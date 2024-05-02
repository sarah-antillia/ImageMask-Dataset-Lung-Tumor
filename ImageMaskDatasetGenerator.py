# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2024/05/03 
# ImageMaskDatasetGenerator.py

import os
import sys
import shutil
import cv2

import glob
import numpy as np
import math
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter

import traceback

# Read file
"""
scan = nib.load('/path/to/stackOfimages.nii.gz')
# Get raw data
scan = scan.get_fdata()
print(scan.shape)
(num, width, height)

"""


"""
def get_mask_boundingbox( mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    ret, bin_img = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
       bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
    points = np.array(contours[0])
    #print(points)
    x, y, w, h = cv2.boundingRect(points)
    rect = (x, y, w, h)
    return rect
"""

class ImageMaskDatasetGenerator:

  def __init__(self, augmentation=True):
    self.W = 512
    self.H = 512

    self.augmentation = augmentation
    if self.augmentation:
      self.hflip    = True
      self.vflip    = False
      self.rotation = True
      self.ANGLES   = [5, 355]
      self.distortion=True
      self.gaussina_filer_rsigma = 40
      self.gaussina_filer_sigma  = 0.5
      self.distortions           = [0.01, 0.02]
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)
      # Don't shrink
      self.resize = False
      self.resize_ratio = 0.8


  def create_mask_files(self, niigz, output_dir, index):
    print("--- create_mask_files {}".format(niigz))
    nii = nib.load(niigz)

    #print("--- nii {}".format(nii))
    data = nii.get_fdata()
    print("---data shape {} ".format(data.shape))
    (width, height, num) = data.shape

    print("--- num_images {}".format(num))
    n = 0
    for i in range(num):
      img = data[:,:,i]
      print("--- shape {}".format(img.shape))
      img = np.array(img)
      img = self.rotate_90(img)
      basename = str(index) + "_" + str(i) + ".jpg"
      filepath = os.path.join(output_dir, basename)
      if img.any() > 0:
        img = img*255
        cv2.imwrite(filepath, img)
        print("Saved {}".format(filepath))
        n += 1
        if self.augmentation:
          self.augment(img, basename, output_dir, border=(0, 0, 0), mask=True)
      else:
        # If mask(img) were all black, skip it. 
        print("---skipped") 
    return n
  
  def create_image_files(self, niigz, output_masks_dir, output_images_dir, index):
    print("--- create_image_files {}".format(niigz))
    nii = nib.load(niigz)

    #print("--- nii {}".format(nii))
    data = nii.get_fdata()
    print("---data shape {} ".format(data.shape))
    (width, height, num) = data.shape

    print("--- num_images {}".format(num))
    n = 0
    for i in range(num):
      img = data[:, :, i]
      img = np.array(img)
      img = self.rotate_90(img)
   
      basename = str(index) + "_" + str(i) + ".jpg"
      mask_filepath = os.path.join(output_masks_dir, basename)
      if os.path.exists(mask_filepath):
        # Save the image file only when the corresponding mask file exists.
        filepath = os.path.join(output_images_dir, basename)
        cv2.imwrite(filepath, img)
        print("Saved {}".format(filepath))
        n += 1
        if self.augmentation:
          self.augment(img, basename, output_images_dir, border=(0, 0, 0), mask=True)
    return n
  

  def generate(self, train_images_dir, train_masks_dir, 
                        output_images_dir, output_masks_dir):

    image_files = glob.glob(train_images_dir + "/lung*.nii.gz")
    mask_files  = glob.glob(train_masks_dir  + "/lung*.nii.gz")
    image_files = sorted(image_files)
    mask_files  = sorted(mask_files)
    num_image_files = len(image_files)
    num_mask_files  = len(mask_files)
    print("--- num_image_files {}".format(num_image_files))
    print("--- num_mask_files  {}".format(num_mask_files))

    index = 10000
    for i, _ in enumerate(mask_files):
      mask_file  = mask_files[i]
      image_file = image_files[i]
      index += 1
      num_masks  = self.create_mask_files(mask_file,   output_masks_dir,  index)
      num_images = self.create_image_files(image_file, output_masks_dir,  output_images_dir, index)
      print(" num_images {}  num_masks: {}".format(num_images, num_masks))
      if num_images != num_masks:
        raise Exception("Num images and segmentations are different ")
      else:
        print("Not found segmentation file {} corresponding to {}".format(mask_file, image_file))

  def rotate_90(self, image, border=(0, 0, 0)):
    h, w = image.shape[:2]
    center = (w/2, h/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(w, h), borderValue=border)
    return rotated_image

  def augment(self, image, basename, output_dir, border=(0, 0, 0), mask=False):
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.rotation:
      self.rotate(image, basename, output_dir, border)

    if self.distortion:
      self.distort(image, basename, output_dir)

    if self.resize:
      self.shrink(image, basename, output_dir, mask)

  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]

  def rotate(self, image, basename, output_dir, border):
    for angle in self.ANGLES:      
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=border)
      output_filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(output_filepath, rotated_image)
      print("--- Saved {}".format(output_filepath))
      
  def distort(self, image, basename, output_dir):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
 
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)
    for size in self.distortions:
      filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + basename
      output_file = os.path.join(output_dir, filename)    
      dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
      dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
      sizex = int(xsize*size)
      sizey = int(xsize*size)
      dx *= sizex/dx.max()
      dy *= sizey/dy.max()

      image = gaussian_filter(image, self.gaussina_filer_sigma)

      yy, xx = np.indices(shape)
      xmap = (xx-dx).astype(np.float32)
      ymap = (yy-dy).astype(np.float32)

      distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
      distorted = cv2.resize(distorted, (w, h))
      cv2.imwrite(output_file, distorted)
      print("=== Saved distorted image file{}".format(output_file))

  def shrink(self, image, basename, output_dir, mask):
    print("----shrink shape {}".format(image.shape))
    h, w    = image.shape[0:2]
    rh = int(h * self.resize_ratio)
    rw = int(w * self.resize_ratio)
    resized = cv2.resize(image, (rw, rh))
    h1, w1  = resized.shape[:2]
    y = int((h - h1)/2)
    x = int((w - w1)/2)
    # black background
    background = np.zeros((w, h, ), np.uint8)
    #if mask == False:
    #  # white background
    #  background = np.ones((h, w, ), np.uint8) * 255
    # paste resized to background
    background[x:x+w1, y:y+h1] = resized
    filename = "shrinked_" + str(self.resize_ratio) + "_" + basename
    output_file = os.path.join(output_dir, filename)    

    cv2.imwrite(output_file, background)
    print("=== Saved shrinked image file{}".format(output_file))

if __name__ == "__main__":
  try:
    train_images_dir  = "./Task06_Lung/imagesTr/"
    train_masks_dir   = "./Task06_Lung/labelsTr"
    output_images_dir = "./Lung-master/images/"
    output_masks_dir  = "./Lung-master/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    # Create jpg image and mask files from nii.gz files under imagesTr and labelsTr.
    generator = ImageMaskDatasetGenerator()
    generator.generate(train_images_dir, train_masks_dir, 
                        output_images_dir, output_masks_dir)
  except:
    traceback.print_exc()


