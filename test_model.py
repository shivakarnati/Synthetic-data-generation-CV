import pickle
import cv2
from utils import get_face_landmarks
import pygame
from pygame.locals import *
import numpy as np
import gc
from mediapipe.python.solutions.face_mesh import FaceMesh
from PIL import Image

class Detect():
    def __init__(self):
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)  
        pygame.display.set_caption("Face Recognition Game")
        self.clock = pygame.time.Clock()
        self.running = False
        self.emotions = ['HAPPY', 'SAD', 'SURPRISED']
        self.camera_index = 0
        self.cap = None
        self.points = 0
        self.font = pygame.font.Font(None, 74)
        self.start_time = 0
        self.game_duration = 30000  # 30 seconds
        self.end_game = False
        
        self.face_mesh = None
        
        with open('./trained_models/model1', 'rb') as f:
            self.model = pickle.load(f)
        
        self.background_image = pygame.image.load('./images/ai3.jpg').convert()
        
        
    def show_start_screen(self):
        # Clear the screen
        #self.screen.fill((0, 0, 0))
        background_image_scaled = pygame.transform.scale(self.background_image, self.screen.get_size())
        
        self.screen.blit(background_image_scaled, (0, 0))
        
        
        title_font = pygame.font.Font(None, 100)
        title_text = title_font.render("Face Recognition Game", True, (255, 255, 0))
        self.screen.blit(title_text, (200, 100))
        
        # Display instructions
        instruction_font = pygame.font.Font(None, 40)
        instruction_text1 = instruction_font.render("Instructions:", True, (0, 255, 127))
        instruction_text2 = instruction_font.render("Look at the camera and make facial expressions to earn points.", True, (255, 255, 255))
        instruction_text3 = instruction_font.render("Happy: +1 point, Sad: -1 point, Surprised: +2 points", True, (255, 255, 255))
        instruction_text4 = instruction_font.render("Tip: If you earn 600 points within 30 seconds, you will win the GAME!!", True, (255, 255, 255))
        instruction_text5 = instruction_font.render("Press any key to start the game.", True, (255, 255, 255))
        
        instruction_text6 = instruction_font.render("Technologies used: ", True, (135, 206, 235))
        instruction_text7 = instruction_font.render("1. Stable Diffusion", True, (255, 255, 255))
        instruction_text8 = instruction_font.render("2. Mediapipe", True, (255, 255, 255))
        instruction_text9 = instruction_font.render("3. Random Forest Classifier", True, (255, 255, 255))


        self.screen.blit(instruction_text1, (200, 250))
        self.screen.blit(instruction_text2, (200, 300))
        self.screen.blit(instruction_text3, (200, 350))
        self.screen.blit(instruction_text4, (200, 400))
        self.screen.blit(instruction_text5, (400, 550))
        
        self.screen.blit(instruction_text6, (200, 700))
        self.screen.blit(instruction_text7, (200, 750))
        self.screen.blit(instruction_text8, (200, 800))
        self.screen.blit(instruction_text9, (200, 850))

        # Update the display
        pygame.display.flip()
    
    def process_frame(self):
        
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            print('Unable to read the camera!')
            return None
        face_landmarks = get_face_landmarks(frame, face_mesh=self.face_mesh, draw=True, static_image_mode=False)
        if face_landmarks:
            output = self.model.predict([face_landmarks])
            emotion = self.emotions[int(output[0])]
            self.update_points(emotion)
            cv2.putText(frame, emotion, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        del face_landmarks
        return frame

    def update_points(self, emotion):
        if emotion == 'HAPPY':
            self.points += 1
        elif emotion == 'SAD':
            self.points -= 1
        elif emotion == 'SURPRISED':
            self.points += 2

    def draw_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = cv2.flip(frame,0)
        frame = cv2.resize(frame, (960, 540))  
        frame_surface = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame_surface, (160, 90)) 

    def draw_points(self):
        points_text = self.font.render(f"Points: {self.points}", True, (0, 255, 255))
        self.screen.blit(points_text, (10, 10))

    def draw_timer(self):
        elapsed_time = pygame.time.get_ticks() - self.start_time
        remaining_time = max(0, (self.game_duration - elapsed_time) // 1000)
        timer_text = self.font.render(f"Time: {remaining_time}", True, (255, 0, 0))
        self.screen.blit(timer_text, (1000, 10))
        if remaining_time == 0:
            self.end_game = True
    
    def show_performance_feedback(self):
        performance_font = pygame.font.Font(None, 30)
        performance_text = performance_font.render(f"Points: {self.points}", True, (255, 255, 255))
        self.screen.blit(performance_text, (10, 10))
        
    def start_game(self):
        self.running = True
        self.end_game = False
        self.points = 0
        self.start_time = pygame.time.get_ticks()
        self.cap = cv2.VideoCapture(self.camera_index)
        self.face_mesh = FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    def end_current_game(self):
        self.running = False
        self.end_game = True
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        if self.face_mesh is not None:
            self.face_mesh.close()
        self.face_mesh = None

    def run(self):
        show_start_screen = True
        
        while True:
            if show_start_screen:
                self.show_start_screen()
                show_start_screen = False
                waiting_for_start = True
                
                while waiting_for_start:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        if event.type == pygame.KEYDOWN:
                            waiting_for_start = False
                            self.start_game()
            
            if self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.end_current_game()
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.end_current_game()
                        show_start_screen = True

                frame = self.process_frame()
                if frame is not None:
                    self.screen.fill((0, 0, 0))  
                    self.draw_frame(frame)
                    self.draw_points()
                    self.draw_timer()
                    pygame.display.flip()
                    del frame
                    gc.collect()
                
                self.clock.tick(40)
                
                if self.end_game:
                    self.end_current_game()
                    result_text = None
                    if self.points >= 600:
                        result_text = self.font.render("Hurray..! You won the Game!!!!", True, (0, 255, 255))
                    else:
                        result_text = self.font.render("You lost the Game! Try again!", True, (255, 0, 0))
                    
                    self.screen.fill((0, 0, 0))
                    self.screen.blit(result_text, (340, 300))
                    game_over_text = self.font.render("Game Over!", True, (255, 0, 0))
                    self.screen.blit(game_over_text, (540, 360))
                    pygame.display.flip()
                    pygame.time.wait(5000)
                    show_start_screen = True

def main():
    detect = Detect()
    detect.run()

if __name__ == '__main__':
    main()
