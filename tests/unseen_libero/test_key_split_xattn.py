import torch

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.key_shared_latent import SharedLatentEncoder
from skill_moe.skillnet.examples.memory_pilot.unseen_libero.key_split_xattn import KeySplitXAttn


def _inputs():
    torch.manual_seed(19)
    return dict(vlm_tokens=torch.randn(2, 11, 12),
                groups=torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, -1, -1, -1],
                                     [0, 1, 1, 2, 2, 3, 3, -1, -1, -1, -1]]),
                history=torch.randn(2, 3, 4),
                history_mask=torch.tensor([[True, True, False], [True, False, False]]),
                state=torch.randn(2, 5))


def _model(gated):
    return KeySplitXAttn(token_dim=12, context_dim=16, token_count=5,
                        history_steps=3, history_action_dim=4, state_dim=5,
                        group_count=6, heads=4, gated_vlm=gated)


def test_split_a_and_b_shape_and_attention_dimensions():
    x = _inputs()
    for gated in (False, True):
        model = _model(gated)
        out, aux = model(**x, return_aux=True)
        assert out.shape == (2, 5, 16)
        torch.testing.assert_close(out.norm(dim=-1), torch.ones(2, 5), atol=1e-6, rtol=1e-6)
        assert aux['history_state_attention'].shape == (2, 4, 5, 4)
        assert aux['vlm_attention'].shape == (2, 4, 5, 11)
        if gated:
            assert aux['gates']['attention'].item() == 0.
            assert aux['gates']['ffn'].item() == 0.


def test_zero_initialized_vlm_gates_make_b_independent_of_vlm_tokens():
    torch.manual_seed(27)
    model = _model(True).eval()
    x = _inputs()
    changed = dict(x, vlm_tokens=x['vlm_tokens'] * -17 + 5)
    with torch.no_grad():
        original = model(**x)
        changed_out = model(**changed)
    torch.testing.assert_close(original, changed_out, atol=0, rtol=0)


def test_a_vlm_path_is_active_and_old_shared_encoder_is_unchanged():
    torch.manual_seed(31)
    a = _model(False).eval()
    x = _inputs()
    changed = dict(x, vlm_tokens=x['vlm_tokens'] + 2.)
    with torch.no_grad():
        assert not torch.allclose(a(**x), a(**changed))

    # The legacy E implementation remains independently constructible and its
    # deterministic forward is unchanged by importing/using the new classes.
    old = SharedLatentEncoder(token_dim=12, context_dim=16, token_count=5,
        history_steps=3, history_action_dim=4, state_dim=5, group_count=6, heads=4)
    old.eval()
    with torch.no_grad():
        before = old(**x)
        _ = a(**x)
        after = old(**x)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
