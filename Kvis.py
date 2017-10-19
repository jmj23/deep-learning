import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


def get_activations(model, inputs):
    activations = []
    inp = model.input
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    if len(inputs.shape) == 3:
        batch_inputs = inputs[np.newaxis, ...]
    else:
        batch_inputs = inputs
    layer_outputs = [func([batch_inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        
    return activations

def show_activations(model,inputs,laynum):
    activs = get_activations(model,inputs)
    layact = activs[laynum]
    numfilt = layact.shape[3]
    numR = int(np.round(np.sqrt(numfilt)))
    numC = int(np.ceil(numfilt/numR))
    fig, ax = plt.subplots(numR, numC)
    fig.set_size_inches(2*numC, 2*numR)
    for rr in range(0,numR):
        for cc in range(0,numC):
            ind = rr*numC + cc
            if ind < layact.shape[3]:
                ax[rr,cc].imshow(layact[0,:,:,ind],cmap='gray')
            
            ax[rr,cc].set_axis_off()
            plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99,
                                wspace=0,hspace=0)
    plt.draw()
    
def visualize_graph(model,filename='model_graph.png',inline=False):
    if inline:
        SVG(model_to_dot(model).create(prog='dot', format='svg'))
    else:
        plot_model(model, to_file=filename)
    