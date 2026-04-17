import math
import sys
import os

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "worldstereo"))


class TestBuildIntrinsics:
    def test_shape(self):
        from nodes.world_stereo import _build_intrinsics
        K = _build_intrinsics(70.0, 768, 480)
        assert K.shape == (3, 3)

    def test_principal_point_at_center(self):
        from nodes.world_stereo import _build_intrinsics
        K = _build_intrinsics(70.0, 768, 480)
        assert K[0, 2].item() == pytest.approx(384.0)
        assert K[1, 2].item() == pytest.approx(240.0)

    def test_focal_from_fov(self):
        from nodes.world_stereo import _build_intrinsics
        K = _build_intrinsics(90.0, 200, 100)
        # fov=90 → fx = (200/2)/tan(45°) = 100
        assert K[0, 0].item() == pytest.approx(100.0, abs=1e-3)


class TestBuildTrajectory:
    def test_circular_shape(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, intrs = _build_trajectory("circular", 16, 1.0, 0.0, 0.0, 70.0, 768, 480)
        assert c2ws.shape == (16, 4, 4)
        assert intrs.shape == (16, 3, 3)

    def test_forward_shape(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, intrs = _build_trajectory("forward", 8, 1.0, 0.5, 0.0, 70.0, 768, 480)
        assert c2ws.shape == (8, 4, 4)

    def test_zoom_in_shape(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, intrs = _build_trajectory("zoom_in", 12, 1.0, 0.0, 0.0, 70.0, 768, 480)
        assert c2ws.shape == (12, 4, 4)

    def test_all_c2ws_are_valid_se3(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, _ = _build_trajectory("circular", 8, 1.0, 0.0, 0.0, 70.0, 768, 480)
        # Bottom row must be [0,0,0,1]
        for i in range(8):
            bottom = c2ws[i, 3, :]
            assert bottom[3].item() == pytest.approx(1.0, abs=1e-5)
            assert bottom[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_intrinsics_replicated_per_frame(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, intrs = _build_trajectory("circular", 5, 1.0, 0.0, 0.0, 70.0, 768, 480)
        # All intrinsics frames must be identical
        for i in range(1, 5):
            assert torch.allclose(intrs[0], intrs[i])


class TestC2WToW2C:
    def test_identity_roundtrip(self):
        from nodes.world_stereo import _c2w_to_w2c
        c2ws = torch.eye(4).unsqueeze(0).repeat(4, 1, 1)
        w2cs = _c2w_to_w2c(c2ws)
        assert torch.allclose(w2cs, c2ws, atol=1e-5)

    def test_shape_preserved(self):
        from nodes.world_stereo import _c2w_to_w2c
        c2ws = torch.randn(10, 4, 4)
        # Make valid SE3
        c2ws[:, 3, :] = torch.tensor([0., 0., 0., 1.])
        w2cs = _c2w_to_w2c(c2ws)
        assert w2cs.shape == (10, 4, 4)
