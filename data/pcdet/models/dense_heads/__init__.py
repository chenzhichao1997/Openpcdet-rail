from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .voxelnext_head import VoxelNeXtHead
from .transfusion_head import TransFusionHead
from .voxelnext_head_ours import VoxelNeXtHead_ours
from .anchor_head_single_ours import AnchorHeadSingle_ours
from .transfusion_head_ours import TransFusionHead_ours
from .voxelnext_head_depth import VoxelNeXtHead_depth
from .bevfusion_TransFusion_Pillar_Head import bevfusion_TransFusion_Pillar_Head
from .sparse_transfusion_head import SparseTransFusionHead
__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'VoxelNeXtHead': VoxelNeXtHead,
    'TransFusionHead': TransFusionHead,
    'VoxelNeXtHead_ours':VoxelNeXtHead_ours,
    'AnchorHeadSingle_ours':AnchorHeadSingle_ours,
    'TransFusionHead_ours':TransFusionHead_ours,
    'VoxelNeXtHead_depth':VoxelNeXtHead_depth,
    'bevfusion_TransFusion_Pillar_Head':bevfusion_TransFusion_Pillar_Head,
    'SparseTransFusionHead':SparseTransFusionHead
}
