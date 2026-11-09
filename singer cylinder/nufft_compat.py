
import torch
from pytorch_finufft.functional import finufft_type1, finufft_type2

if int(torch.__version__.split('.')[0]) < 2:
    from pytorch_finufft.functional import FinufftType1, FinufftType2

    class _Type1(torch.autograd.Function):
        @staticmethod
        def forward(ctx, points, values, shape, options):
            result = FinufftType1.forward(points, values, shape, options)
            FinufftType1.setup_context(ctx, (points, values, shape, options), result)
            return result

        @staticmethod
        def backward(ctx, grad):
            return FinufftType1.backward(ctx, grad)[:4]

    class _Type2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, points, values, options):
            result = FinufftType2.forward(points, values, options)
            FinufftType2.setup_context(ctx, (points, values, options), result)
            return result

        @staticmethod
        def backward(ctx, grad):
            return FinufftType2.backward(ctx, grad)[:3]

    def finufft_type1(points, values, output_shape, **kwargs):
        return _Type1.apply(points, values, output_shape, kwargs)

    def finufft_type2(points, values, **kwargs):
        return _Type2.apply(points, values, kwargs)
