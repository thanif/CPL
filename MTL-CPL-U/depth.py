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

try:
    from PIL import PILLOW_VERSION
except:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)

try:
    import PIL.Image as Image
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

# Numpy for datastructures
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)


class CDepth(object):
    cam = None
    dispNp = None

    def setCamera(self, cam):
        self.cam = cam

    # read the 16bit disparity image
    def readFromDisparityImage(self, imgFile):
        dispImg = Image.open(imgFile)
        self.dispNp = np.array(dispImg, dtype=np.float)

        # convert to correct disparity again
        self.dispNp = (self.dispNp - 1.) / 256.

    # calculates the distance in camera coordinates based on the given disparity
    def getDistanceFromDisparity(self, disparity):
        if not self.cam:
            print("No camera object. Use setCamera() first")
            return -1
        distance = self.cam.intrinsic.fx * self.cam.extrinsic.baseline / disparity
        return distance
