import pygame
import numpy as np
import time
import random

# play a simple rhythm game that has left right up down arrows that need to be pressed at a certain time
def playGame():
    # display left up right down arrows
    crashed = False

    # default bag of arrows
    defaultArrowBag = [0,1,2,3]
    arrowBag = []
    arrow = None

    gameStart = False
    # up arrow
    up = pygame.image.load('arrow.png')
    up = pygame.transform.scale(up, (100,100))
    # down arrow
    down = pygame.image.load('arrow.png')
    down = pygame.transform.rotate(down, 180)
    down = pygame.transform.scale(down, (100,100))
    # left arrow
    left = pygame.image.load('arrow.png')
    left = pygame.transform.rotate(left, 90)
    left = pygame.transform.scale(left, (100,100))
    # right arrow
    right = pygame.image.load('arrow.png')
    right = pygame.transform.rotate(right, 270)
    right = pygame.transform.scale(right, (100,100))

    score = 0
    fallingArrows = []
    startTime = 0
    arrowNeeded = False
    arrowsAdded = 0
    while crashed is not True:
        dt = clock.tick(60)
        if gameStart:
            startTime += dt
        prevArrow = arrow
        # if arrow bag is empty, reset it
        if len(arrowBag) == 0:
            # copy default bag to arrowBag
            arrowBag = list(defaultArrowBag)
            random.shuffle(arrowBag)
        
        if arrow is None:
            arrowNeeded = True
            # pick a random arrow from the bag and remove it from the bag
            nextArrow = arrowBag.pop()
            if nextArrow == prevArrow:
                nextArrow = arrowBag.pop()
            arrow = nextArrow

        # display the score
        largeText = pygame.font.Font('freesansbold.ttf',35)
        TextSurf = largeText.render("Score: "+str(score), True, black)
        TextRect = TextSurf.get_rect()
        TextRect.center = ((display_width/2),(display_height/2)-50)
        display.blit(TextSurf, TextRect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if arrow == 0:
                        # is there a falling arrow in the green box?
                        if len(fallingArrows) > 0:
                            for fallingArrow in fallingArrows:
                                if fallingArrow['type'] == 0 and fallingArrow['green']:
                                    score += 1
                                    fallingArrow['shouldRemove'] = True
                                    break
                elif event.key == pygame.K_DOWN:
                    # is there a falling arrow in the green box?
                        if len(fallingArrows) > 0:
                            for fallingArrow in fallingArrows:
                                if fallingArrow['type'] == 1 and fallingArrow['green']:
                                    score += 1
                                    fallingArrow['shouldRemove'] = True
                                    break
                elif event.key == pygame.K_LEFT:
                     # is there a falling arrow in the green box?
                        if len(fallingArrows) > 0:
                            for fallingArrow in fallingArrows:
                                if fallingArrow['type'] == 2 and fallingArrow['green']:
                                    score += 1
                                    fallingArrow['shouldRemove'] = True
                                    break
                elif event.key == pygame.K_RIGHT:
                     # is there a falling arrow in the green box?
                        if len(fallingArrows) > 0:
                            for fallingArrow in fallingArrows:
                                if fallingArrow['type'] == 3 and fallingArrow['green']:
                                    score += 1
                                    fallingArrow['shouldRemove'] = True
                                    break
                elif event.key == pygame.K_SPACE:
                    gameStart = True
        
       
        if gameStart:
            if score < -5:
                crashed = True
            
            # display the arrow
            if arrowNeeded:
                arrowsAdded += 1
                if arrow == 0:
                    img = up.copy()
                    fallingArrow = {'img':img, 'x':(display_width/2-50), 'y':-100, 'type':0, 'id':arrowsAdded, 'shouldRemove': False, 'green': False}
                    fallingArrows.append(fallingArrow)
                    arrowNeeded = False
                    # display.blit(up, (display_width/2-50, display_height/2-50))
                elif arrow == 1:
                    img = down.copy()
                    fallingArrow = {'img':img, 'x':(display_width/2-50), 'y':-100, 'type':1, 'id':arrowsAdded, 'shouldRemove': False, 'green': False}
                    fallingArrows.append(fallingArrow)
                    arrowNeeded = False
                    # display.blit(down, (display_width/2-50, display_height/2-50))
                elif arrow == 2:
                    img = left.copy()
                    fallingArrow = {'img':img, 'x':(display_width/2-50) - 200, 'y':-100, 'type':2, 'id':arrowsAdded, 'shouldRemove': False, 'green': False}
                    fallingArrows.append(fallingArrow)
                    arrowNeeded = False
                    # display.blit(left, (display_width/2-50, display_height/2-50))
                elif arrow == 3:
                    img = right.copy()
                    fallingArrow = {'img':img, 'x':(display_width/2-50) + 200, 'y':-100, 'type':3, 'id':arrowsAdded, 'shouldRemove': False, 'green': False}
                    fallingArrows.append(fallingArrow)
                    arrowNeeded = False
                    # display.blit(right, (display_width/2-50, display_height/2-50))
            
            

             # clear the screen 
            display.fill((0,0,0))
            pygame.display.set_caption("Rhythm Game"+"  "+"SCORE: "+str(score))

            # draw the green zone near the bottom
            pygame.draw.rect(display, green, (0,display_height-250,display_width,150))
            
            # draw the falling arrows
            for fallingArrow in fallingArrows:
                display.blit(fallingArrow['img'], (fallingArrow['x'], fallingArrow['y']))
                fallingArrow['y'] += 0.05 * dt
                # is the falling arrow in the green zone?
                if fallingArrow['y'] > (display_height-250) and fallingArrow['y'] < (display_height-150):
                    print(fallingArrow['type'])
                    fallingArrow['green'] = True
                

                if fallingArrow['shouldRemove']:
                    fallingArrows.remove(fallingArrow)
                elif fallingArrow['y'] > display_height:
                    fallingArrows.remove(fallingArrow)
                    score = score - 1
           
            pygame.display.update()

            # every 2 seconds, add a new arrow to the bag
            if startTime > 2000:
                startTime = 0
                arrow = None
        
    return score
    # # display the arrows on the screen
    # display.blit(up, (50,50))
    # display.blit(down, (50,250))
    # display.blit(left, (50,150))
    # display.blit(right, (250,150))


def displayWait():
    largeText = pygame.font.Font('freesansbold.ttf',35)
    TextSurf = largeText.render("Press Space to Play", True, black)
    TextRect = TextSurf.get_rect()
    TextRect.center = ((display_width/2),(display_height/2))
    display.blit(TextSurf, TextRect)
    # display small text under
    smallText = pygame.font.Font('freesansbold.ttf',20)
    TextSurf = smallText.render("Calibrate webcam detector first", True, black)
    TextRect = TextSurf.get_rect()
    TextRect.center = ((display_width/2),(display_height/2)+50)
    display.blit(TextSurf, TextRect)

   

if __name__ == "__main__":
    
    ###### initialize required parameters ########  
    display_width = 500
    display_height = 500
    green = (0,255,0)
    red = (255,0,0)
    black = (0,0,0)
    window_color = (200,200,200)
    clock=pygame.time.Clock() 

    score = 0
    
    pygame.init() #initialize pygame modules    

    #### display game window #####

    display = pygame.display.set_mode((display_width,display_height))
    display.fill(window_color)
    displayWait()
    pygame.display.update()
    
    score = playGame()
    print("Final Score: "+str(score))
    pygame.quit()