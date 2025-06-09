from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE, DynamicPillarVFESimple2D, FineGrainedPFE
from .dynamic_voxel_vfe import DynamicVoxelVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .dynamic_pillar_vfe_PointNet import DynamicPillarVFESimple2DMax
__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'DynamicPillarVFESimple2D': DynamicPillarVFESimple2D,
    'DynamicVoxelVFE': DynamicVoxelVFE,
    'DynamicPillarVFESimple2DMax':DynamicPillarVFESimple2DMax,
    'FineGrainedPFE':FineGrainedPFE
}
