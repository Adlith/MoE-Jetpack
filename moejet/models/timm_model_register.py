from timm.models._registry import register_model
from .vit_timm import _create_vision_transformer
from .vit_softmoe_timm import _create_vision_transformer_moe

# @register_model
# def vit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     """ ViT-Small (ViT-S/16)
#     """
#     model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
#     model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model

@register_model
def vit_tiny_custom(pretrained=False, **kwargs): #5,543,716
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_tiny_moe_custom(pretrained=False, **kwargs): #5,543,716
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer_moe('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_small_custom(pretrained=False, **kwargs): #5,543,716
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_moe_custom(pretrained=False, **kwargs): #5,543,716
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer_moe('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model