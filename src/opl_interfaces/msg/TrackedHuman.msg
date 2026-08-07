int32 track_id          # Persistent Identity ID (assigned by IdentityManager)
int32 raw_track_id      # Raw tracking ID (assigned by ByteTrack)
string name             # Recognized name or "Person_X"
float32 confidence      # Detection confidence
float32 match_score     # Cosine similarity score
geometry_msgs/Point position_3d
sensor_msgs/RegionOfInterest bbox
geometry_msgs/Point[] landmarks
