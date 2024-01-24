import numpy as np
import matplotlib.pyplot as plt

class Cushion:
    def __init__(self, point1, point2):
        self.points = np.array([point1, point2])
        self.middle = (self.points[0] + self.points[1]) / 2
        self.length = np.hypot(*(self.points[1] - self.points[0]))

class Pocket:
    def __init__(self, position, diameter):
        self.position = np.array(position)
        self.diameter = diameter

class Ball:
    def __init__(self, position, color, diameter, mass):
        self.position = np.array(position)
        self.color = color.lower()
        self.diameter = diameter
        self.mass = mass

        self.position_history = []
        self.velocity = np.array([0, 0])
    
    def move(self, new_position):
        self.position_history.append(self.position)
        self.position = new_position

class Simulation:
    # Creation
    def __init__(self, cue_ball_position, table_type="english_pool_7ft"):
        valid_table_types = ["english_pool_7ft"]
        
        if not table_type in valid_table_types:
            raise Exception(f"{table_type} is not a valid table type")
        
        self.table_type = table_type
        self._initiate_table(table_type)
        self.cue_ball = self._create_ball(cue_ball_position, "white", True)
        self.balls = []

    def _initiate_table(self, table_type):
        self.cushions = []
        self.pockets = []

        if table_type == "english_pool_7ft":
            self.size = np.array([914, 1830]) # Dimensions from farthest part of cushion (where cushion meets wood) - mm

            cushion_depth = 50.8 # Cushion size = 2" = 50.8mm
            pocket_diameter = 100 # A rough estimate

            # Cushions
            self.cushions.append(Cushion([cushion_depth + pocket_diameter/2, cushion_depth],
                                         [self.size[0] - cushion_depth - pocket_diameter/2, cushion_depth])) # Bottom
            self.cushions.append(Cushion([self.size[0] - cushion_depth, cushion_depth + pocket_diameter/2],
                                         [self.size[0] - cushion_depth, self.size[1]/2 - pocket_diameter/2]))
            self.cushions.append(Cushion([self.size[0] - cushion_depth, self.size[1]/2 + pocket_diameter/2],
                                         [self.size[0] - cushion_depth, self.size[1] - cushion_depth - pocket_diameter/2]))
            self.cushions.append(Cushion([self.size[0] - cushion_depth - pocket_diameter/2, self.size[1] - cushion_depth],
                                         [cushion_depth + pocket_diameter/2, self.size[1] - cushion_depth]))
            self.cushions.append(Cushion([cushion_depth, cushion_depth + pocket_diameter/2],
                                         [cushion_depth, self.size[1]/2 - pocket_diameter/2]))
            self.cushions.append(Cushion([cushion_depth, self.size[1]/2 + pocket_diameter/2],
                                         [cushion_depth, self.size[1] - cushion_depth - pocket_diameter/2]))

            # Pockets
            self.pockets.append(Pocket([cushion_depth, cushion_depth], pocket_diameter))
            self.pockets.append(Pocket([cushion_depth, self.size[1]/2], pocket_diameter))
            self.pockets.append(Pocket([cushion_depth, self.size[1] - cushion_depth], pocket_diameter))
            self.pockets.append(Pocket([self.size[0] - cushion_depth, cushion_depth], pocket_diameter))
            self.pockets.append(Pocket([self.size[0] - cushion_depth, self.size[1]/2], pocket_diameter))
            self.pockets.append(Pocket([self.size[0] - cushion_depth, self.size[1] - cushion_depth], pocket_diameter))

    def _create_ball(self, position, color, is_cue=False):
        if self.table_type == "english_pool_7ft":
            if is_cue:
                diameter = 47.6 # 1 7/8"
                density = 1720 # kg/m^3 (Super Aramith Pro English 8 Ball)
                mass = 0.097 
            else:
                diameter = 50.8 # 2"
                mass = 0.118 # kg (4.15oz)

        return Ball(position, color, diameter, mass)

    def add_ball(self, position, color):
        self.balls.append(self._create_ball(position, color))

    # Drawing
    def _draw_cushions(self):
        for cushion in self.cushions:
            plt.plot(
                [cushion.points[0][0], cushion.points[1][0]],
                [cushion.points[0][1], cushion.points[1][1]],
                color="blue"
            )
    
    def _draw_pockets(self):
        for pocket in self.pockets:
            circle = plt.Circle(
                np.array(pocket.position),
                radius=pocket.diameter/2,
                color="red",
                fill=False
            )

            plt.gca().add_patch(circle)
    
    def _draw_balls(self):
        for ball in self.balls + [self.cue_ball]:
            # if ball.color == "white":
            #     circle = plt.Circle(
            #         np.array(ball.position),
            #         radius=ball.diameter/2,
            #         color="black",
            #         fill=False
            #     )
            
            # else:
            #     circle = plt.Circle(
            #         np.array(ball.position),
            #         radius=ball.diameter/2,
            #         color=ball.color,
            #         fill=True
            #     )

            # plt.gca().add_patch(circle)
            
            circle = plt.Circle(
                np.array(ball.position),
                radius=ball.diameter/2,
                color=ball.color,
                fill=True
            )
            plt.gca().add_patch(circle)
            
            circle = plt.Circle(
                np.array(ball.position),
                radius=ball.diameter/2,
                color="black",
                fill=False
            )
            plt.gca().add_patch(circle)
    
    def draw(self):
        plt.figure()

        plt.xlim(0, self.size[0])
        plt.ylim(0, self.size[1])
        plt.gca().set_aspect('equal')

        self._draw_cushions()
        self._draw_pockets()
        self._draw_balls()

        plt.show()

    # Simulation
    def simulate(self, cue_direction, cue_power):
        pass

if __name__ == "__main__":
    sim = Simulation([300, 300])

    sim.add_ball([200, 200], "red")
    sim.add_ball([300, 200], "yellow")

    sim.draw()