from simplenn import Network
from simplenn.optimizers import RMSProp


def test_network():
    network = Network(optimizer=RMSProp())
    print(network)
