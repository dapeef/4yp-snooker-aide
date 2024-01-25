#! /usr/bin/env python

import pooltool as pt
from attr import asdict
import numpy as np

def english_8_ball_table_specs() -> pt.PocketTableSpecs:
    return pt.PocketTableSpecs(
        l = 1.9812,
        w = 1.9812 / 2,
        cushion_width = 2 * 2.54 / 100,
        cushion_height = 0.64 * 2 * 0.028575,
        corner_pocket_width = 0.10,
        corner_pocket_angle = 1,
        corner_pocket_depth = 0.0398,
        corner_pocket_radius = 0.124 / 2,
        corner_jaw_radius = 0.08,
        side_pocket_width = 0.08,
        side_pocket_angle = 3,
        side_pocket_depth = 0.03,
        side_pocket_radius = 0.129 / 2,
        side_jaw_radius = 0.03,
    )

def ball_params(is_cue_ball=False) -> pt.BallParams:
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
        **asdict(ball_params(is_cue_ball))
    )

def main():
    # We need a table, some balls, and a cue stick
    # Build a table from the table specs
    table = pt.Table.from_table_specs(english_8_ball_table_specs())
    # balls = pt.get_eight_ball_rack(table)

    balls = {
        "cue" : create_ball("cue", (0.5, .5), True)
    }

    positions = [(ball_params().R, ball_params().R),
                 (.2, .2),
                 (.4, .4),
                 (.6, .6),
                 (.8, .8),
                 (.9, .8),
                 (.8, .9),
                 (.8, 1),
                 (.8, 1.1),
                 (.8, 1.2),
                 (.8, 1.3),
                 (1.9812/2 - ball_params().R, 1.9812 - ball_params().R)
                ]

    for i in range(len(positions)):
        # ball_id = "red_" + ("0" if i <= 9 else "") + str(i + 1)
        # balls[ball_id] = pt.Ball.create(
        #     ball_id, xy=positions[i], ballset=pt.BallSet(name="generic_snooker"), **asdict(pt.BallParams.default())
        # )

        ball_id = str(i + 1)
        balls[ball_id] = create_ball(ball_id, positions[i])

    cue = pt.Cue(cue_ball_id="cue")

    # Wrap it up as a System
    shot = pt.System(table=table, balls=balls, cue=cue)

    # Aim at the head ball with a strong impact
    shot.strike(V0=2, phi=pt.aim.at_ball(shot, "2") + 2.7, a=0.5, b=-.3, theta=0)
    # shot.strike(V0=0)

    # Evolve the shot.
    pt.simulate(shot, inplace=True)

    shot = pt.continuize(shot)

    shot.save("temp/pool_tool_output.json")

    # for state in shot.balls['cue'].history_cts.states:
    #     print(state)



    # # Start an instance of the ShotViewer
    # interface = pt.ShotViewer()

    # # Open up the shot in the GUI
    # interface.show(shot)

if __name__ == "__main__":
    main()