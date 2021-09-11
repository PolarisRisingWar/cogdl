from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

def test_Github():
    args = build_args_from_dict({"dataset": "rd2cd_Github"})
    data = build_dataset(args)
    assert data.data.num_nodes == 37700
    assert data.data.num_edges == 289003
    assert data.num_features == 4005
    assert data.num_classes == 2


if __name__ == "__main__":
    test_Github()