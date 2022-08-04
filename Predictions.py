import pickle
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

model = pickle.load(open('img_model.p','rb'))
Categories = ["Healthy", "Diseased"]
dir = input("Enter the path:")
img = imread(dir)
#plt.imshow(img)
#plt.show()
img_resize=resize(img,(150,150,3))
l=[img_resize.flatten()]
probability=model.predict_proba(l)
for ind,val in enumerate(Categories):
  print(val,"=",probability[0][ind]*100,'%')
print("The predicted image is : "+Categories[model.predict(l)[0]])
