import task_fire_world
from task_fire_world import TaskFireWorld

# define tasks
task_three_ways = TaskFireWorld(order='C', nrows=9, ncols=9,
                                start_location=0,
                                fire_locations=[12, 13, 14, 15, 16, 22, 32, 42, 47, 57, 58],
                                wall_locations=[70, 69],
                                goal_locations=[17],
                                task_name='task_three_ways')

# [[  nan  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]
# [ -0.1  -0.1  -0.1 -10.  -10.  -10.  -10.  -10.    5. ]
# [ -0.1  -0.1  -0.1  -0.1 -10.   -0.1  -0.1  -0.1  -0.1]
# [ -0.1  -0.1  -0.1  -0.1  -0.1 -10.   -0.1  -0.1  -0.1]
# [ -0.1  -0.1  -0.1  -0.1  -0.1  -0.1 -10.   -0.1  -0.1]
# [ -0.1  -0.1 -10.   -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]
# [ -0.1  -0.1  -0.1 -10.  -10.   -0.1  -0.1  -0.1  -0.1]
# [ -0.1  -0.1  -0.1  -0.1  -0.1  -0.1   0.    0.   -0.1]
# [ -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]]

task_maze = TaskFireWorld(order='C', nrows=10, ncols=10,
                          start_location=0,
                          fire_locations=[21, 22, 32, 36, 37, 40, 42, 43, 50, 61, 62, 63],
                          wall_locations=[11, 12, 13, 14, 15, 16, 17, 18, 38, 48, 58, 68, 78, 86, 87, 88],
                          goal_locations=[54],
                          task_name='task_maze')
# [[  nan  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]
#  [ -0.1   0.    0.    0.    0.    0.    0.    0.    0.   -0.1]
#  [ -0.1 -10.  -10.   -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]
#  [ -0.1  -0.1 -10.   -0.1  -0.1  -0.1 -10.  -10.    0.   -0.1]
#  [-10.   -0.1 -10.  -10.   -0.1  -0.1  -0.1  -0.1   0.   -0.1]
#  [-10.   -0.1  -0.1  -0.1   5.   -0.1  -0.1  -0.1   0.   -0.1]
#  [ -0.1 -10.  -10.  -10.   -0.1  -0.1  -0.1  -0.1   0.   -0.1]
#  [ -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1   0.   -0.1]
#  [ -0.1  -0.1  -0.1  -0.1  -0.1  -0.1   0.    0.    0.   -0.1]
#  [ -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]]


task_fireplaces = TaskFireWorld(order='C', nrows=10, ncols=10,
                                start_location=0,
                                fire_locations=[21, 23, 26, 27, 33, 45, 47, 52, 56, 58, 62, 64, 74, 84, 85],
                                # wall_locations=[11,12,13,14,15,16,17,18,38,48,58,68,78,86,87,88],
                                goal_locations=[77],
                                task_name='task_fireplaces')

# [[  nan  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]
# [ -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]
# [ -0.1 -10.   -0.1 -10.   -0.1  -0.1 -10.  -10.   -0.1  -0.1]
# [ -0.1  -0.1  -0.1 -10.   -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]
# [ -0.1  -0.1  -0.1  -0.1  -0.1 -10.   -0.1 -10.   -0.1  -0.1]
# [ -0.1  -0.1 -10.   -0.1  -0.1  -0.1 -10.   -0.1 -10.   -0.1]
# [ -0.1  -0.1 -10.   -0.1 -10.   -0.1  -0.1  -0.1  -0.1  -0.1]
# [ -0.1  -0.1  -0.1  -0.1 -10.   -0.1  -0.1   5.   -0.1  -0.1]
# [ -0.1  -0.1  -0.1  -0.1 -10.  -10.   -0.1  -0.1  -0.1  -0.1]
# [ -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1]]

task_five = TaskFireWorld(order='C', nrows=10, ncols=10,
                          start_location=0,
                          fire_locations=[13, 14, 31, 33, 36, 43, 45, 46, 62, 63, 66],
                          wall_locations=[15, 16, 18, 41, 51, 61, 71, 81, 83, 84, 85, 86, 87],
                          goal_locations=[55],
                          task_name='task_five')

task_six = TaskFireWorld(order='C', nrows=10, ncols=10,
                         start_location=0,
                         fire_locations=[22, 27, 32, 40, 47, 50, 54, 64, 67, 71, 81],
                         wall_locations=[83, 84, 85, 86, 87],
                         goal_locations=[90],
                         task_name='task_six')

task_mini = TaskFireWorld(order='C', nrows=3, ncols=3,
                          start_location=0,
                          fire_locations=[2, 7],
                          goal_locations=[8],
                          task_name='task_mini')


task_fireplaces_2 = TaskFireWorld(order='C', nrows=10, ncols=10,
                                start_location=0,
                                fire_locations=[13, 23, 33, 21, 31, 41, 62, 72, 54, 64, 92, 93, 94, 35, 36, 37, 76],
                                wall_locations=[48, 58, 68, 78, 88],
                                goal_locations=[56],
                                task_name='task_fireplaces_2')

task_five_ways = TaskFireWorld(order='C', nrows=10, ncols=10,
                         start_location=40,
                         fire_locations=[11, 17, 36, 37, 45, 46, 47, 52, 65, 66, 68 ,88],
                         wall_locations=[12, 13, 14, 15, 16, 18, 21, 22, 31, 32, 34, 35, 44, 51, 62, 63, 64, 78,
                                         81, 82, 83, 84, 85, 86, 87],
                         goal_locations=[59],
                         task_name='task_five_ways')

task_straight = TaskFireWorld(order='C', nrows=10, ncols=10,
                         start_location=40,
                         fire_locations=[11, 12, 32, 52, 36, 56, 37, 57, 86, 87],
                         wall_locations=[11, 21, 31, 51, 61, 71, 81, 33,
                                         15, 25, 35, 55, 65, 75, 85, 17, 27, 67, 77, 53, 63, 73, 83, 82, 16, 13, 23],
                         goal_locations=[48],
                         task_name='task_straight')


if __name__ == '__main__':
    print(task_five)
