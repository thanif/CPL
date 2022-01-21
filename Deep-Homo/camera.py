#!/usr/bin/python
#   ----------------------
#   The Tsinghua-Daimler Cyclist Benchmark
#   ----------------------
#
#
#   License agreement
# -  ----------------
#
#   This dataset is made freely available for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use, copy, and distribute the data given that you agree:
#   1. That the dataset comes "AS IS", without express or implied warranty. Although every effort has been made to ensure accuracy, Daimler (or the website host) does not accept any responsibility for errors or omissions.
#   2. That you include a reference to the above publication in any published work that makes use of the dataset.
#   3. That if you have altered the content of the dataset or created derivative work, prominent notices are made so that any recipients know that they are not receiving the original data.
#   4. That you may not use or distribute the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
#   5. That this original license notice is retained with all copies or derivatives of the dataset.
#   6. That all rights not expressly granted to you are reserved by Daimler.
#
#   Contact
#   -------
#
#   Fabian Flohr
#   mail: tdcb at fabian-flohr.de


# Python imports
import os
import sys
import json
import math

# depth package
try:
    import depth
except:
    print("Failed to import depth package.")
    sys.exit(-1)

# json for datastructures
try:
    import json
except:
    print("Failed to import json package.")
    sys.exit(-1)


def printError(message):
    print('ERROR: ', message)
    print('\n')
    sys.exit(-1)


class CIntrinsic(object):
    def __init__(self):
        self.fx = -1.
        self.fy = -1.
        self.u0 = -1
        self.v0 = -1.


class CExtrinsic(object):
    def __init__(self):
        self.baseline = -1.
        self.x = -1.
        self.y = -1.
        self.z = -1.
        self.yaw = -1.
        self.pitch = -1.
        self.roll = -1.


class CIsoCamera(object):

    def __init__(self):
        self.intrinsic = CIntrinsic()
        self.extrinsic = CExtrinsic()
        self.initialized = False

    def loadFromJson(self, filename):
        with open(filename, 'r') as f:
            jsonText = f.read()
            jsonDict = json.loads(jsonText)
            for key in jsonDict["intrinsic"]:
                if key in self.intrinsic.__dict__:
                    self.intrinsic.__dict__[key] = jsonDict["intrinsic"][key]
            for key in jsonDict["extrinsic"]:
                if key in self.extrinsic.__dict__:
                    self.extrinsic.__dict__[key] = jsonDict["extrinsic"][key]
            self.initialized = True

    def image_to_camera(self, u, v, disparity):
        if (not self.initialized):
            printError("Camera must be correctly initialized first.")

        if (disparity <= 0):
            return [0, 0, 0]

        # Compute 3D point in camera coordinate system
        # supposing standard oordinate conventions: x (lon), y (lat), z (height)
        xCam = (self.intrinsic.fx * self.extrinsic.baseline) / disparity
        yCam = - (xCam / self.intrinsic.fx) * (u - self.intrinsic.u0)
        zCam = (xCam / self.intrinsic.fy) * (self.intrinsic.v0 - v)

        return [xCam, yCam, zCam]

    def image_to_world(self, u, v, disparity):
        if (not self.initialized):
            printError("Camera must be correctly initialized first.")

        if (disparity <= 0):
            return [0, 0, 0]

        [xCam, yCam, zCam] = self.image_to_camera(u, v, disparity)

        # Correct with camera position and tilt angle to get the vehicle coordinate system (mid rear axis, street level)
        # supposing standard oordinate conventions: x (lon), y (lat), z (height)
        yWorld = yCam + self.extrinsic.y
        xWorld = xCam * math.cos(self.extrinsic.pitch) + zCam * \
            math.sin(self.extrinsic.pitch) + self.extrinsic.x
        zWorld = - xCam * math.sin(self.extrinsic.pitch) + zCam * \
            math.cos(self.extrinsic.pitch) + self.extrinsic.z

        return [xWorld, yWorld, zWorld]

    def getDistanceFromDisparity(self, disparity):
        if (not self.initialized):
            printError("Camera must be correctly initialized first.")
        if (disparity <= 0):
            return 0
        distance = self.intrinsic.fx * self.extrinsic.baseline / disparity
        return distance


# The main method, if you execute this script directly
def main(argv):

    # Example of how to use this
    cam = CIsoCamera()
    cam.loadFromJson(
        "[ROOT]/camera/train/tsinghuaDaimlerDataset/tsinghuaDaimlerDataset_2015-03-24_041424_000028651_camera.json")
    cam_vector = cam.image_to_camera(500, 500, 30)
    print("x: "+str(cam_vector[0]) + "  y: " +
          str(cam_vector[1])+"  z: "+str(cam_vector[2]))
    world_vector = cam.image_to_world(500, 500, 30)
    print("x: "+str(world_vector[0]) + "  y: " +
          str(world_vector[1])+"  z: "+str(world_vector[2]))

    # Example of how to use depth loading
    dep = depth.CDepth()
    dep.setCamera(cam)
    dep.readFromDisparityImage(
        "[ROOT]/disparity/train/tsinghuaDaimlerDataset/tsinghuaDaimlerDataset_2015-03-24_041424_000028651_disparity.png")
    dep.getDistanceFromDisparity(30)


if __name__ == "__main__":
    main(sys.argv[1:])
