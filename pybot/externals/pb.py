from pybot.externals.proto import vs_pb2 as vs

def pose_t(orientation, pos):
    p = vs.pose_t()
    p.orientation.extend(orientation)
    p.pos.extend(pos)
    return p

def serialize(msg):
    return msg.SerializeToString()

def deserialize(msg):
    return ParseFromString(msg)
