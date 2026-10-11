from app.core.security import create_token, decode_token, hash_password, verify_password


def test_password_hash_roundtrip():
    hashed = hash_password("rahasia123")
    assert hashed != "rahasia123"
    assert verify_password(hashed, "rahasia123")
    assert not verify_password(hashed, "salah")
    assert not verify_password("bukan-hash", "rahasia123")


def test_token_kind_is_enforced():
    access = create_token("user-1", "access")
    assert decode_token(access, "access") == "user-1"
    assert decode_token(access, "refresh") is None
    assert decode_token("token.palsu.sekali", "access") is None
