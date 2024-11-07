# Code from module 1

from IPython.display import display_html
from typing import List, Tuple, Dict, Callable
from copy import deepcopy
import math
import heapq
import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np


full_world = [
    ['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰'], 
    ['🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '⛰'], 
    ['🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌋', '🌾', '🌾', '🌾', '⛰', '🌾', '🌾', '⛰', '⛰', '🌋', '🌾'], 
    ['🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '⛰', '⛰', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌋', '🌋', '🌋', '🌋', '🌾', '⛰', '🌾', '🌾', '🌾', '⛰', '🌋', '🌾'], 
    ['🌾', '🌾', '🌋', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🐊', '🐊', '🐊', '🐊', '🐊', '🌋', '🌋', '🌲', '🌋', '🌋', '🌋', '⛰', '⛰', '🌾', '🌾', '⛰', '⛰', '🌾'], 
    ['🌲', '🌾', '🌋', '🌋', '🌋', '🌋', '⛰', '⛰', '⛰', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌋', '🌲', '🌲', '🌋', '🌋', '⛰', '⛰', '🌾', '⛰', '⛰', '🌾', '🌾'], 
    ['🌲', '🌾', '🌲', '🌋', '🌋', '⛰', '⛰', '⛰', '🌾', '🌾', '🐊', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾'], 
    ['🌲', '🌲', '🌲', '🌋', '🌲', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '⛰', '⛰', '🌾', '🌾', '🌾'], 
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '⛰', '⛰', '🌾', '🌾', '🌾'], 
    ['🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'], 
    ['🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '⛰', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌾', '🌾', '⛰', '🌾'], 
    ['🌲', '🌲', '🌲', '🌲', '🐊', '🌾', '⛰', '🌾', '🌾', '🌋', '🌋', '⛰', '⛰', '🌾', '⛰', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌾', '🌋', '⛰', '🌾'], 
    ['🌲', '🌲', '🌲', '🐊', '🐊', '🐊', '🌋', '🌾', '⛰', '🌋', '🌋', '🌾', '🌾', '🌾', '⛰', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌾', '🌋', '🌋', '⛰'], 
    ['🌲', '🌲', '🌲', '🐊', '🐊', '🐊', '🌋', '⛰', '⛰', '🌋', '⛰', '🌾', '🐊', '🌾', '⛰', '🌋', '🌲', '🌲', '🌲', '🌾', '🌾', '🌋', '🌾', '🌾', '🌋', '🌋', '⛰'], 
    ['🌲', '🌲', '🌲', '🌲', '🐊', '🐊', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🐊', '🌾', '⛰', '🌋', '🌋', '🌲', '🌾', '🌾', '🌋', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰'], 
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🐊', '🐊', '🌋', '🌋', '🌾', '🌾', '🌾', '🐊', '🐊', '🌾', '⛰', '🌋', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '⛰'], 
    ['🌾', '🌾', '🌲', '🌲', '🌲', '🐊', '🐊', '🌋', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '⛰', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰', '⛰'], 
    ['🌾', '🌾', '🌋', '🌲', '🌲', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🐊', '🌾', '🐊', '🐊', '🌾', '🌾', '🌋', '⛰', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾'], 
    ['🌾', '🌋', '🌋', '🌲', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌋', '⛰', '🌾', '🌾', '🌾', '⛰', '🌋', '🌾', '⛰', '🌾'], 
    ['🌾', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🐊', '🌾', '🌾', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾'], 
    ['🌾', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰', '⛰', '🌾', '🌾', '⛰', '🌋', '🌾', '🌾', '🌾', '🐊', '🐊', '🌾', '⛰', '🌋', '🌋', '🌾'], 
    ['🌾', '🌋', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌋', '⛰', '⛰', '🌋', '🌋', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '⛰', '🌋', '⛰'], 
    ['🌾', '🌋', '⛰', '⛰', '🌋', '⛰', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '⛰', '🌾', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '⛰', '🌋', '⛰'], 
    ['🌾', '🌋', '🌋', '🌋', '🌋', '🌋', '⛰', '⛰', '⛰', '🌾', '⛰', '⛰', '🌾', '🌾', '⛰', '⛰', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '⛰'], 
    ['🌾', '🌋', '🌋', '🌋', '🌋', '⛰', '🌾', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾'], 
    ['🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾'], 
    ['🌾', '🌾', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾']
]

small_world = [
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾']
]

small_world_test = [
    ['🌾', '🌲', '🌲', '🐊', '🌲','🌋', '🌲'],
    ['🌾', '🌲', '🌲', '🌲', '⛰','🌲', '🌲'],
    ['🌾', '🌲', '🐊', '🌲', '🌲', '🌲', '🌲']
    ]

def display_emoji_grid(emoji_grid):
    """
    Display a List of Lists of emojis in a perfect grid (table) in streamlit.
    
    Parameters:
    emoji_grid (list of list of str): A 2D list containing emojis to display in a grid.
    """
    html = '<table style="border-collapse: collapse; text-align: center; border:none;">'
    
    for row in emoji_grid:
        html += '<tr>'
        for emoji in row:
            html += f'<td style="padding: 1px; font-size: 1em;">{emoji}</td>'
        html += '</tr>'
    
    html += '</table>'
    
    st.markdown(html, unsafe_allow_html=True)



def get_succesors(current_location: Tuple[int, int], world:List[List], move_map:List[Tuple[int,int]], cost_map:Dict) -> List[Tuple[Tuple[int,int],int]]:
    """
    get_succesors: takes a location and checks the possible successors of that location in the given world. Will return all the possible moves with their associated cost 
    current_location: Tuple[int,int] -  coordinate in the current world grid, we will evaluate the tiles around it in this order -> (DOWN, RIGHT, UP, LEFT)
    world: List[List[str]]: - the actual context for the navigation problem.
    move_map: :List[Tuple[int,int]] - contains how we can traverse the **world**
    cost_map:  Dict[str, int] - is a `Dict` of costs for each type of terrain in **world**.
    """
    future_moves = []
    max_x = len(world[0])
    max_y = len(world)

    for move in move_map:
        x = current_location[0] + move[0]
        y = current_location[1] + move[1]

        if 0 <= x < max_x and 0 <= y < max_y:
            cost = cost_map[world[y][x]]
            future_moves.append(((x,y),cost))
        else: continue
    return future_moves


def heuristic(current_position:Tuple[int,int], goal:Tuple[int,int], type_alg:int): # you can add formal parameters
    """
    The heuristic function should return a value that represent the cost of the path to the goal for the given position

    current_position: Tuple[int,int] - the position to evaulate the heuristic on
    goal: Tuple[int,int] - final position, used to evaluate the heuristic against the **current_location**
    type_alg: int - chosses the type of distance equation to use, **0** for manhattan distance and **1** for euclidean distance
    """
    if type_alg == 0:
        return 1* (abs(current_position[0] - goal[0]) + abs(current_position[1] - goal[1]))
    elif type_alg == 1:
        return math.sqrt((current_position[0] - goal[0])**2 + (current_position[1] - goal[1])**2)
    

def reconstruct_path(came_from: Dict[Tuple[int, int], Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Args:
        came_from (Dict[Tuple[int, int], Tuple[int, int]]): A dictionary mapping each node to the node it came from.
        start (Tuple[int, int]): The starting node represented as a coordinate tuple (x, y).
        goal (Tuple[int, int]): The goal node represented as a coordinate tuple (x, y).

    Returns:
        List[Tuple[int, int]]: A list of nodes representing the path from the start node to the goal node, inclusive.
    """
    path = []
    current = goal
    while current is not start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def reconstruct_moves(came_from: Dict[Tuple[int, int], Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Args:
        came_from (Dict[Tuple[int, int], Tuple[int, int]]): A dictionary mapping each node to the node it came from.
        start (Tuple[int, int]): The starting node represented as a coordinate tuple (x, y).
        goal (Tuple[int, int]): The goal node represented as a coordinate tuple (x, y).

    Returns:
        List[Tuple[int, int]]: A list of moves represented as coordinate differences (dx, dy), indicating the steps taken from the start node to reach the goal node.
    """
    
    moves = []
    current = goal

    while current is not start:
        previous = came_from[current]
        move = (current[0] - previous[0], current[1] - previous[1])
        moves.append(move)
        current = previous

    moves.reverse()  
    return moves
    


def a_star_search( world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
    """

    a_start_search implementation to traverse the given 2D world based on the cost of the terrain. It utilizes the following equation $f(n) = g(n) + h(n)$ where g(n) is the total cost of the path and the h(n) is the results of the heuristic function which is an estimate of the cost from n position to the goal.  

    world List[List[str]]: the actual context for the navigation problem.
    start Tuple[int, int]: the starting location of the bot, `(x, y)`.
    goalTuple[int, int]: the desired goal position for the bot, `(x, y)`.
    costs Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.
    moves List[Tuple[int, int]]: the legal movement model expressed in offsets in **world**.
    heuristic Callable: is a heuristic function, $h(n)$.


    **returns** List[Tuple[int, int]]: the offsets needed to get from start state to the goal as a `List`.
    """
    frontier = []
    heapq.heappush(frontier, (0, start))

    total_cost = {start: 0}
    best_path = {start:None}
    explored = set()
    while frontier: 
        _, current_position = heapq.heappop(frontier)

        if current_position == goal: 
            return reconstruct_moves(best_path, start, goal)
        
        explored.add(current_position)

        successors = get_succesors(current_position, world, moves, costs)

        for possible_move, cost in successors:
            if possible_move in explored:
                continue
            
            possible_move_cost = total_cost[current_position] + cost

            if possible_move not in total_cost or possible_move_cost < total_cost[possible_move]:
                total_cost[possible_move] = possible_move_cost
                f_n = possible_move_cost + heuristic(possible_move, goal, 1) # using euclidean distance for heuristic
                heapq.heappush(frontier, (f_n,possible_move))
                best_path[possible_move] = current_position

    return [] # if no path is possible

def pretty_print_path( world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    
    """
    *pretty_print_path* will display the world and then the path it took to get to the goal. It will return the cost of the displayed path

    * **world** List[List[str]]: the world (terrain map) for the path to be printed upon.
    * **path** List[Tuple[int, int]]: the path from start to goal, in offsets.
    * **start** Tuple[int, int]: the starting location for the path.
    * **goal** Tuple[int, int]: the goal location for the path.
    * **costs** Dict[str, int]: the costs for each action.

    **returns** int - The path cost.
    """
    
    
    copy_world = deepcopy(world)
    
    direction_emojis = {
        (1, 0): '⏩',
        (-1, 0): '⏪',
        (0, 1): '⏬',
        (0, -1): '⏫'
    }
    #display_emoji_grid(copy_world)
    value_cost = 0
    x, y = start[0], start[1]
    for move in path: 
        emoji = world[y][x]
        value_cost += costs[emoji]
        copy_world[y][x] = direction_emojis[move]
        x += move[0]
        y += move[1]
    copy_world[goal[1]][goal[0]] = '🎁'
    display_emoji_grid(copy_world)
    return value_cost # replace with the real value!


# Moves and Cost Dictionaries
MOVES = [(0,-1), (1,0), (0,1), (-1,0)]
COSTS = { '🌾': 1, '🌲': 3, '⛰': 5, '🐊': 7,'🌋':500}

# Code for module 4
def get_world_size(world):
    return (len(world[0]), len(world))

world_options = {
    f"World 1 {get_world_size(small_world_test)}": small_world_test,
    f"World 2 {get_world_size(small_world)}": small_world,
    f"World 3 {get_world_size(full_world)}": full_world
}



# Reset function
def reset_inputs():
    st.session_state.start_x = 0
    st.session_state.start_y = 0
    st.session_state.goal_x = 1
    st.session_state.goal_y = 1
    st.session_state.world_choice = list(world_options.keys())[0]

# Initialize session state if not already done
if 'start_x' not in st.session_state:
    st.session_state.start_x = 0
    st.session_state.start_y = 0
    st.session_state.goal_x = 1
    st.session_state.goal_y = 1
    st.session_state.world_choice = list(world_options.keys())[0]

# Let the user choose between 3 different worlds
st.sidebar.header("Select a world, start and goal coordinates to run A* algorithm")
world_choice = st.sidebar.selectbox("Choose your world:", options=list(world_options.keys()), index=0)
selected_world = world_options[world_choice]
size_x, size_y = get_world_size(selected_world)
# UI SideBar Controls
st.sidebar.header("Select Start and Goal Points")
# Start Coordinate Selection
column1, column2 = st.sidebar.columns(2)
with column1:
    start_x = st.number_input("Start X", min_value=0, max_value=size_x -1,value=0, step=1)

with column2: 
    start_y = st.number_input("Start Y", min_value=0,max_value=size_y - 1,value=0, step=1)

# Goal Coordinate Selection
column3, column4 = st.sidebar.columns(2)
with column3:
    goal_x = st.number_input("Goal X", min_value=0,max_value=size_x - 1,value=1, step=1)

with column4:
    goal_y = st.number_input("Goal Y", min_value=0,max_value=size_y - 1,value=1, step=1)


column5, column6 = st.sidebar.columns(2)
with column6:
    # Button for running A*
    run_button = st.button('Run A*')
with column5:
    # Reset button
    if st.button('Reset'):
        reset_inputs()

# For user friendliness, show the starting point and the goal point on the world
display_world = []
for row in selected_world:
    display_world.append(row.copy())

# Handlig errors + indicators
if 0 <= start_x < size_x and 0 <= start_y < size_y:
    if display_world[start_y][start_x] == '🌋':
        display_world[start_y][start_x] = '❌'
    else: 
        display_world[start_y][start_x] = '🔷'  # Starting point

if 0 <= goal_x < size_x and 0 <= goal_y < size_y:
    if display_world[goal_y][goal_x] == '🌋':
        display_world[goal_y][goal_x] = '❌'
    else:
        display_world[goal_y][goal_x] = '🎁'  # Goal point

if display_world[start_y][start_x] == display_world[goal_y][goal_x]:
    display_world[goal_y][goal_x] = '❌'
    display_world[start_y][start_x] = '❌'
# Display Selected World
if not run_button:
    st.markdown("# Selected World:")
    display_emoji_grid(display_world)

# if choosing to run A*
if run_button:
    if not ( (0 <= start_x < size_x and 0 <= start_y < size_y) and (0 <= goal_x < size_x and 0 <= goal_y < size_y)):
        st.error("Error: Invalid start or goal coordinates!")
    elif display_world[goal_y][goal_x] ==  '❌' or display_world[start_y][start_x] == '❌':
        st.error("Error: You cannot start or end in a volcano! Please select a valid coordinate")
    elif display_world[goal_y][goal_x] == display_world[start_y][start_x]:
        st.warning("You selected the same coordinate for start and goal, please select different coordinates")
    else: 
        path = a_star_search(selected_world, (start_x,start_y),(goal_x,goal_y), COSTS, MOVES, heuristic)
        st.markdown('# Path Taken:')
        value_cost = pretty_print_path(selected_world, path, (start_x,start_y),(goal_x,goal_y), COSTS)
        st.markdown(f'### Movement Cost: {value_cost}')