from collections import namedtuple

__all__ = [
    'Options',
    'DEFAULT_OPTIONS',
]

Options = namedtuple('Options', [
    'am_out_c_dim',
    'nm_out_c_dim',
    'mask_out_c_dim',
    'gf_dim',
    'lighting_model',
])

DEFAULT_OPTIONS = Options(
    am_out_c_dim=3,
    nm_out_c_dim=2,
    mask_out_c_dim=1,
    gf_dim=64,
    lighting_model='pretrained/illu_pca',
)
