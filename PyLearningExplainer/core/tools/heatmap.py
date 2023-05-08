
import matplotlib.pyplot as pyplot
import numpy

class HeatMapImages():
  
  def __init__(self, size, title):
    self.size = size
    self.features = []
    self.images = []
    self.title = title

  def set_instance(self, instance):
    self.images.append(numpy.zeros(self.size))
    self.images[-1] = numpy.reshape(instance, self.size)

  def add_features(self, features):
    self.images.append([numpy.zeros(self.size),numpy.zeros(self.size)])
    images = self.images[-1]
    with_weights = all(weight is not None for (_,_,_,weight) in features)
    if with_weights:
      max_weights = max(weight for (_,_,_,weight) in features if weight != 0)
      min_weights = min(weight for (_,_,_,weight) in features if weight != 0)
      
    for (feature, threshold, sign, weight) in features:
      x = (feature - 1) // self.size[0]
      y = (feature - 1) % self.size[1]
      color = (weight / (max_weights-min_weights)) * 256 if with_weights else 256
      if sign is True:
        images[0][x][y] = color
      else:
        images[1][x][y] = color

    images[0] = numpy.ma.masked_where(images[0] < 0.9, images[0])
    images[1] = numpy.ma.masked_where(images[1] < 0.9, images[1])

class HeatMap():

  def __init__(self, x, y, instance=None):
    self.size = (x, y)
    self.heat_map_images = [] 
    
    
  '''
  Do not forget to convert an implicant in features thanks to tree.to_features().
  Create two images per reason. This one positive and this one negative. 
  '''  
  def new_image(self, title):
    self.heat_map_images.append(HeatMapImages(self.size, title))
    return self.heat_map_images[-1] 

  def display(self):
    n_images = len(self.heat_map_images)
    fig, axes = pyplot.subplots(1, n_images, figsize=self.size)

    for i, heat_map_images in enumerate(self.heat_map_images):
      axes.flat[i].title.set_text(heat_map_images.title)
      for image in heat_map_images.images:
        if isinstance(image, list):
          axes.flat[i].imshow(image[0], alpha=0.6, cmap='Blues', vmin = 0, vmax = 255, interpolation='None')
          axes.flat[i].imshow(image[1], alpha=0.6, cmap='Reds', vmin = 0, vmax = 255, interpolation='None')
        else:
          axes.flat[i].imshow(image)
    pyplot.show()

  def display_observation(self):
    pyplot.imshow(self.image_observation) 
    pyplot.show()




  