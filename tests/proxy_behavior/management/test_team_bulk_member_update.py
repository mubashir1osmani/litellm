import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# PATCH /v2/team/{team_id}/members — actor x team-shape matrix. Same gates as
# /team/member_update (proxy admin, the team's team admin, or an org admin of
# the team's org; else 403 in the handler), pinned against the bulk route.
_MATRIX = [
    ("alpha/proxy_admin", Actor.PROXY_ADMIN, "alpha", 200),
    ("alpha/org_admin", Actor.ORG_ADMIN, "alpha", 200),
    ("alpha/team_admin", Actor.TEAM_ADMIN, "alpha", 200),
    ("alpha/internal_user", Actor.INTERNAL_USER, "alpha", 403),
    ("alpha/owner", Actor.OWNER, "alpha", 403),
    ("alpha/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "alpha", 403),
    ("alpha/cross_org_user", Actor.CROSS_ORG_USER, "alpha", 403),
    ("alpha/service_account", Actor.SERVICE_ACCOUNT, "alpha", 403),
    ("alpha/org_b_admin", Actor.ORG_B_ADMIN, "alpha", 403),
    ("beta/proxy_admin", Actor.PROXY_ADMIN, "beta", 200),
    ("beta/org_admin", Actor.ORG_ADMIN, "beta", 403),
    ("beta/team_admin", Actor.TEAM_ADMIN, "beta", 403),
    ("beta/internal_user", Actor.INTERNAL_USER, "beta", 403),
    ("beta/owner", Actor.OWNER, "beta", 403),
    ("beta/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "beta", 403),
    ("beta/cross_org_user", Actor.CROSS_ORG_USER, "beta", 403),
    ("beta/service_account", Actor.SERVICE_ACCOUNT, "beta", 403),
    ("beta/org_b_admin", Actor.ORG_B_ADMIN, "beta", 200),
]


async def _seed_target(prisma, world, shape: str, team_id: str, member_ids: list) -> None:
    if shape == "alpha":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_a_id,
            admin_user_ids=[world.keys[Actor.TEAM_ADMIN].user_id],
            member_user_ids=member_ids,
        )
    elif shape == "beta":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_b_id,
            member_user_ids=member_ids,
        )
    else:  # pragma: no cover - guard
        pytest.fail(f"unknown shape={shape}")


def _role_of(row, user_id: str):
    for m in row.members_with_roles or []:
        if m["user_id"] == user_id:
            return m["role"]
    return None


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_bulk_member_update_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    member_id = scratch.tag("member")
    await _seed_target(prisma, world, shape, scratch.prefix, [member_id])
    caller = world.keys[actor]

    resp = await proxy_client.patch(
        f"/v2/team/{scratch.prefix}/members",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"user_ids": [member_id], "update_fields": {"role": "admin"}},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert _role_of(row, member_id) == "admin"
    else:
        assert _role_of(row, member_id) == "user", "denied but role changed"


async def test_team_bulk_member_update_reports_non_members_and_skips_unselected(
    proxy_client, prisma, scratch, world
):
    """A mixed user_ids list updates only the selected member: the non-member
    lands in failed_updates and an unselected member keeps its role."""
    selected = scratch.tag("m1")
    unselected = scratch.tag("m2")
    outsider = scratch.tag("not-a-member")
    await _seed_target(prisma, world, "alpha", scratch.prefix, [selected, unselected])

    resp = await proxy_client.patch(
        f"/v2/team/{scratch.prefix}/members",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"user_ids": [selected, outsider], "update_fields": {"role": "admin"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_requested"] == 2
    assert [u["user_id"] for u in body["successful_updates"]] == [selected]
    assert [f["user_id"] for f in body["failed_updates"]] == [outsider]

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    assert _role_of(row, selected) == "admin"
    assert _role_of(row, unselected) == "user"
    assert _role_of(row, outsider) is None


async def test_team_bulk_member_update_all_members_in_team(
    proxy_client, prisma, scratch, world
):
    """all_members_in_team=True updates every member without listing user_ids."""
    m1 = scratch.tag("m1")
    m2 = scratch.tag("m2")
    await _seed_target(prisma, world, "beta", scratch.prefix, [m1, m2])

    resp = await proxy_client.patch(
        f"/v2/team/{scratch.prefix}/members",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"all_members_in_team": True, "update_fields": {"role": "admin"}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["total_requested"] == 2

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    assert _role_of(row, m1) == "admin"
    assert _role_of(row, m2) == "admin"


async def test_team_bulk_member_update_over_max_batch_is_400(
    proxy_client, prisma, scratch, world
):
    """A user_ids list larger than the 500-member cap is rejected 400."""
    await _seed_target(prisma, world, "alpha", scratch.prefix, [scratch.tag("m1")])
    resp = await proxy_client.patch(
        f"/v2/team/{scratch.prefix}/members",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={
            "user_ids": [f"{scratch.prefix}-u{i}" for i in range(501)],
            "update_fields": {"role": "admin"},
        },
    )
    assert resp.status_code == 400, resp.text


async def test_team_bulk_member_update_unknown_team_is_400(
    proxy_client, prisma, scratch, world
):
    resp = await proxy_client.patch(
        f"/v2/team/{scratch.prefix}-missing/members",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"user_ids": [scratch.tag("m")], "update_fields": {"role": "admin"}},
    )
    assert resp.status_code == 400, resp.text
