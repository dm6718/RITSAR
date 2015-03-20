##############################################################################
#                                                                            #
#  This program enables a user to inscribe data from a signal's spectrum.    #
#  The user clicks and drags from the top-left to bottom-right and the       #
#  corner points are output.  OpenCV is required.                            #
#                                                                            #
##############################################################################

import cv2

#Define behavior for mouse callback function
def onMouse(event,x,y, flags, param):
    pos1    = param[0]
    pos2    = param[1]
    img_new = param[2]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = (x,y)
        pos1     = param[0]
        
    elif event == cv2.EVENT_LBUTTONUP:
        param[1] = (x,y)
        pos2     = param[1]
        img_out = img_new[pos1[1]:pos2[1],
                          pos1[0]:pos2[0]]
        cv2.imshow('win', img_out)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        img_out = img_new
        cv2.imshow('win', img_out)

def inscribe(img):
    print('\
Click and drag from top-left to bottom-right\n\
to inscribe phase history \n\n\
Right-click to reset image\n\
Press enter when finished\
         ')
    
    #Scale intensity values for an 8-bit display
    img_new = img-img.min()
    img_new = img_new/img_new.max()
    
    #Confortably size window for a 1920 x 1080 display
    rows = int(1080/2)
    cols = int(1920/2)    
    
    cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('win', cols, rows)
    
    #Install mouse callback
    pos1 = (0,0)
    pos2 = (-1,-1)
    params = [pos1, pos2, img_new]
    cv2.setMouseCallback('win',onMouse, param = params)
    
    #Display image
    cv2.imshow('win',img_new)
    
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            cv2.destroyAllWindows()
            break

    return(params[0:2])   