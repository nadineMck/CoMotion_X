from comotion_x.perception.model import model_sha256, verify_pose_model


def test_pose_model_hash_verification_rejects_wrong_file(tmp_path) -> None:
    path = tmp_path / "model.task"
    path.write_bytes(b"not a model")

    assert len(model_sha256(path)) == 64
    assert not verify_pose_model(path)

