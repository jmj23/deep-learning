import pygame
import time
import numpy as np

s = 300
screen = pygame.display.set_mode((s, s))

screenarray = np.zeros((s,s))
start = time.time()
for x in range(100):
    screenarray.fill(np.random.randint(0,256))
    pygame.surfarray.blit_array(screen, screenarray)
    pygame.display.flip()
print(100./(time.time()-start))
   
input()