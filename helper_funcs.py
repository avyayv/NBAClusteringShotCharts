import requests
import json
from sklearn.cluster import KMeans
import umap
import numpy as np
import requests
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Circle, Rectangle, Arc

cluster_names = ["Trans. Big Man", "Moreyball 3pt spec", "Midrange", "Trad. Big Man", "Moreyball paint spec"]
locations = {
    "Mid-Range": ['L', 'R', 'C', 'RC', 'LC'], 
    "Above the Break 3": ['C', 'RC', 'LC'], 
    "Restricted Area": ['C'], 
    "Corner 3": ['R', 'L'], 
    "In The Paint (Non-RA)": ['R', 'L','C']
}

def get_stats_from_scid(typeof, scid):
    playerid, year = scid.split('-')
    filename = typeof+'/'+year+'.json'
    loaded = json.loads(open(filename).read())
    headers = loaded['resultSets'][0]['headers']
    for player in loaded['resultSets'][0]['rowSet']:
        if str(player[0]) == (playerid):
            player_data = player
    player_data = dict(zip(headers, player_data))
    return player_data

#written by http://savvastjortjoglou.com/nba-shot-sharts.html
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def get_post_processed_data(shots):
    shot_charts = {}
    shot_chart_raw = {}
    for shot in shots:
        day = shot[-3]
        first_part = day[2:4]
        int_year = int(day[2:4])
        if int_year == 0:
            int_year = 100
        second_part = str(int_year-1)[-2:]
        if len(second_part) == 1:
            second_part = "0"+second_part
        overall = second_part+first_part
        if day[4] == '1':
            second_part = str(int_year+1)[-2:]
            if len(second_part) == 1:
                second_part = "0"+second_part
            overall = first_part+second_part
        shot_chart_id = str(shot[3])+'-'+overall
        made = 0
        if 'Made' in shot[10]:
            made = 1
        if shot_chart_id in shot_charts:
            shot_charts[shot_chart_id].append([shot[13], shot[14], made])
            shot_chart_raw[shot_chart_id].append(np.array([int(shot[17]), int(shot[18])]))
        else:
            shot_charts[shot_chart_id] = [[shot[13], shot[14], made]]
            shot_chart_raw[shot_chart_id] = [np.array([int(shot[17]), int(shot[18])])]
            
    before_umap_shot_chart_data = []
    for key in list(shot_charts.keys()):
        current_number = 0
        fgm_fga_per_loc = np.zeros([14,2])
        freq_fgp_per_loc = np.zeros([14,1])
        for location_large in (list(locations.keys())):
            for location_small in (list(locations[location_large])):
                for shot in shot_charts[key]:
                    if location_large in shot[0] and location_small in shot[1]:
                        fgm_fga_per_loc[current_number][1] += 1
                        if shot[2] == 1:
                            fgm_fga_per_loc[current_number][0] += 1
                current_number += 1

        for idx, loc in enumerate(fgm_fga_per_loc):
            if np.sum(fgm_fga_per_loc[:,1]) > 0:
                freq_fgp_per_loc[idx][0] = (loc[1])/(np.sum(fgm_fga_per_loc[:,1]))
        before_umap_shot_chart_data.append(freq_fgp_per_loc.flatten())
        
    return shot_chart_raw, before_umap_shot_chart_data