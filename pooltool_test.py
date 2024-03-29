#! /usr/bin/env python

import pooltool as pt
from attr import asdict
import numpy as np
import time

def english_8_ball_table_specs() -> pt.PocketTableSpecs:
    return pt.PocketTableSpecs(
        l = 1.594,
        w = 0.821,
        cushion_width = 0.041,
        cushion_height = 0.031, # 0.037 to very top
        corner_pocket_width = 0.1, # 10cm
        corner_pocket_angle = 23 / 2, # need protractor
        corner_pocket_depth = 0.055,
        corner_pocket_radius = 0.06,
        corner_jaw_radius = 0.063,
        side_pocket_width = 0.14,
        side_pocket_angle = 80/2,
        side_pocket_depth = 0.042,
        side_pocket_radius = 0.045,
        side_jaw_radius = 0.042,
    )

def english_8_ball_ball_params(is_cue_ball=False) -> pt.BallParams:
    density = 1720 # kg/mm^3 (Super Aramith Pro English 8 Ball)

    if is_cue_ball:
        radius =  47.6 / 2 / 1000 # m - diameter = 1 7/8" = 47.6mm
    else:
        radius = 50.8 / 2 / 1000 # m - diameter = 2"

    mass = 4/3 * np.pi * radius**3 * density # kg

    # Overwrite radius so all balls are the same size; pooltool can't handle different sized balls at the moment
    radius = 50.8 / 2 / 1000 # m - diameter = 2"

    return pt.BallParams(
        m=mass,
        R=radius,
        u_s=0.2,
        u_r=0.01,
        u_sp_proportionality=10 * 2 / 5 / 9,
        e_c=0.85,
        f_c=0.2,
        g=9.81,
    )

def create_ball(id, position, is_cue_ball=False) -> pt.Ball:
    if is_cue_ball:
        id = "cue"
    
    return pt.Ball.create(
        id,
        xy=position,
        ballset=pt.BallSet(name="pooltool_pocket"),
        **asdict(english_8_ball_ball_params(is_cue_ball))
    )

def get_example_balls():
    balls = {
        "cue" : create_ball("cue", (0.5, 1), is_cue_ball=True)
    }

    positions = [(english_8_ball_ball_params().R, english_8_ball_ball_params().R),
                 (.2, .2),
                 (.4, .4),
                 (.55, .55),
                 (.7, .8),
                 (.7, .6),
                 (.7, .9),
                 (.7, 1),
                 (.7, 1.1),
                 (.7, 1.2),
                 (.7, 1.3),
                 (1.9812/2 - english_8_ball_ball_params().R, 1.9812 - english_8_ball_ball_params().R)
                ]

    for i in range(len(positions)):
        ball_id = str(i + 1)
        balls[ball_id] = create_ball(ball_id, positions[i])
    
    return balls

def main():
    # We need a table, some balls, and a cue stick
    # Build a table from the table specs
    table = pt.Table.from_table_specs(english_8_ball_table_specs())
    # balls = pt.get_eight_ball_rack(table)

    cue = pt.Cue(cue_ball_id="cue")
    
    balls = get_example_balls()

    # Wrap it up as a System
    shot = pt.System(table=table, balls=balls, cue=cue)

    # Aim at the head ball with a strong impact
    start = time.time()
    for i in range(360):
        shot.strike(V0=2, phi=pt.aim.at_ball(shot, "2") + i + .1, a=0.5, b=-.3, theta=0)
        # Evolve the shot.
        simulated_shot = pt.simulate(shot)
        
        shot = pt.continuize(simulated_shot)
    # print(time.time() - start)
    # shot.strike(V0=0)


    shot = pt.continuize(simulated_shot)

    shot.save("temp/pool_tool_output.json")

    # for state in shot.balls['cue'].history_cts.states:
    #     print(state)



    # # Start an instance of the ShotViewer
    # interface = pt.ShotViewer()

    # # Open up the shot in the GUI
    # interface.show(shot)

if __name__ == "__main__":
    main()