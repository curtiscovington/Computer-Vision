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
    while crashed is not True:
        prevArrow = arrow
        # if arrow bag is empty, reset it
        if len(arrowBag) == 0:
            # copy default bag to arrowBag
            arrowBag = list(defaultArrowBag)
            random.shuffle(arrowBag)
        
        if arrow is None:
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
                        score += 1
                        arrow = None
                    else:
                        score -= 1
                        arrow = None
                elif event.key == pygame.K_DOWN:
                    if arrow == 1:
                        score += 1
                        arrow = None
                    else:
                        score -= 1
                        arrow = None
                elif event.key == pygame.K_LEFT:
                    if arrow == 2:
                        score += 1
                        arrow = None
                    else:
                        score -= 1
                        arrow = None
                elif event.key == pygame.K_RIGHT:
                    if arrow == 3:
                        score += 1
                        arrow = None
                    else:
                        score -= 1
                        arrow = None
                elif event.key == pygame.K_SPACE:
                    gameStart = True
        
        if gameStart:
            display.fill(window_color)
            # display the arrow
            if arrow == 0:
                display.blit(up, (display_width/2-50, display_height/2-50))
            elif arrow == 1:
                display.blit(down, (display_width/2-50, display_height/2-50))
            elif arrow == 2:
                display.blit(left, (display_width/2-50, display_height/2-50))
            elif arrow == 3:
                display.blit(right, (display_width/2-50, display_height/2-50))
            pygame.display.set_caption("Rhythm Game"+"  "+"SCORE: "+str(score))
            pygame.display.update()
            

        clock.tick(2)
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
    arrow = pygame.image.load('arrow.png')
    clock=pygame.time.Clock() 

    score = 0
    
    pygame.init() #initialize pygame modules    

    #### display game window #####

    display = pygame.display.set_mode((display_width,display_height))
    display.fill(window_color)
    displayWait()
    pygame.display.update()
    
    final_score = playGame()

    pygame.quit()