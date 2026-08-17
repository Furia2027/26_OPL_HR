# Human Recognition

A ROS 2 workspace that functions as human recognition system

## Documentation

For more information, check the documentation link below:
https://docs.google.com/document/d/1AXH329UweO3HN9avNHV6fKshVtfs3GVqtQvc72vNd1s/edit?usp=sharing

### Dependencies

* Ubuntu Linux 22.04 Jammy / 24.04 Nobble
* ROS 2 Humble / Jazzy

### Installation for Packages

* Check Documentation

### Executing program

* Remember to source the workspace first
```
source install/setup.bash
```
* Launching the whole system
```
ros2 launch opl_human_vision test_detector.launch.py
```
* Assign Name for tracked person (Temporary Testing)
```
ros2 service call /face_recognizer_node/enroll_person opl_interfaces/srv/EnrollPerson "{raw_track_id: 1, name: 'Lucas'}"
```

## Authors

Furia2027
[@Furia2027](https://discord.com/users/furia_1001)

## Version History

* Jazzy
    * OS Upgrade
* Humble
    * Initial Release

## License

## Acknowledgments
