import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


CIFAR10_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


class Display:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = len(x)
        self.index = self._gen_rand_index()
        self.im = plt.imshow(self.x[self.index])
        self.title_text = self._get_title_from_index()
        self.title_obj = plt.title(self.title_text)
        
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.next_button = Button(axnext, 'Next')
        self.next_button.on_clicked(self.next_btn_clicked)

        plt.show()  
            
    def _gen_rand_index(self):
        return int(random.random()*self.size)

    def _get_title_from_index(self):
        title_index = self.y[self.index][0]
        title_text = CIFAR10_CLASSES[title_index]
        return title_text

    def next_btn_clicked(self, event):
        self.index = self._gen_rand_index()
        self.im.set_data(self.x[self.index])
        self.title_text = self._get_title_from_index()
        self.title_obj.set_text(self.title_text)
    