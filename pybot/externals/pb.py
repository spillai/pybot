from pybot.externals.proto import vs_pb2 as vs

def serialize(msg):
    return msg.SerializeToString()

def deserialize(msg):
    return ParseFromString(msg)
