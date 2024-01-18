#! /usr/bin/env python

import pooltool as pt
from attr import asdict

# We need a table, some balls, and a cue stick
table = pt.Table.default(pt.TableType.SNOOKER)
# balls = pt.get_eight_ball_rack(table)

balls = {
    "cue" : pt.Ball.create(
        "cue", xy=(.5 , 0.5), ballset=pt.BallSet(name="generic_snooker"), **asdict(pt.BallParams.default())
    )
}

positions = [(0.5, 0.5),
             (1, 0.5),
             (1, 1),
             (1, 2)]

for i in range(4):
    ball_id = "red_" + ("0" if i <= 9 else "") + str(i + 1)
    balls[ball_id] = pt.Ball.create(
        ball_id, xy=positions[i], ballset=pt.BallSet(name="generic_snooker"), **asdict(pt.BallParams.default())
    )

cue = pt.Cue(cue_ball_id="cue")

# Wrap it up as a System
shot = pt.System(table=table, balls=balls, cue=cue)

# Aim at the head ball with a strong impact
shot.strike(V0=1.5, phi=pt.aim.at_ball(shot, "red_02") + .6, a=0, b=.8, theta=0)

# Evolve the shot.
pt.simulate(shot, inplace=True)

shot = pt.continuize(shot)

shot.save("temp/pool_tool_output.json")

for state in shot.balls['cue'].history_cts.states:
    print(state)



# Start an instance of the ShotViewer
interface = pt.ShotViewer()

# Open up the shot in the GUI
interface.show(shot)