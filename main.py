import numpy as np

# Defining management policy.
VSP = 1
PAG = 2
SEG = 3

# Defining fit algorithms.
FIRST_FIT = 1
BEST_FIT = 2
WORST_FIT = 3

memory = np.array(0)
free_memory = 0
page_table = np.zeros(0)

# A list of tuples of format: (Start of segment, Size of segment, Process ID, Segment number)
segment_table = []

s_space = "        "
d_space = "                "


# Class for holding the information for a process
class Process:
    process_id = 0
    arrival_time = 0
    lifetime = 0
    address_space = []
    admitted_time = 0
    block_indexes = []
    pages = []

    def __init__(self, the_id, the_times, space):
        self.process_id = the_id
        self.arrival_time = the_times[0]
        self.lifetime = the_times[1]
        self.address_space = space
        self.pages = [0]

    def __str__(self):
        return str(self.process_id)


# Reading the input file to the system and returning a list of all the processes.
def read_file():
    global memory
    global free_memory

    file_name = input("Input File: ")
    file = open(file_name, "r")
    process_size = int(file.readline())

    free_memory = process_size

    processes = []

    # Reading in all the processes from the file.
    for i in range(0, process_size):
        tmp = file.readline().strip()
        process_id = int(tmp)
        times = [int(x) for x in file.readline().strip().split(" ")]
        address_spaces = [int(x) for x in file.readline().strip().split(" ")]
        address_spaces.pop(0)

        # Adding the process to the end of the list.
        process = Process(process_id, times, address_spaces)
        processes.append(process)
        file.readline()

    # Sorting the list by the arrival time of the process.
    processes.sort(key=lambda x: x.arrival_time)  # reverse=True

    return processes


# Running the variable space partitioning algorithm to place processes in memory.
def vsp(fit_algorithm, process):
    global memory
    fit_hole = 0

    # Finding the starting and ending indexes of all the holes in memory.
    holes = []
    if memory[0] == 0:
        holes.append(0)

    for i in range(1, len(memory) - 1):
        if (memory[i - 1] != 0 and memory[i] == 0) or (memory[i] == 0 and memory[i + 1] != 0):
            holes.append(i)

    if memory[len(memory) - 1] == 0:
        holes.append(len(memory) - 1)

    holes = np.array(holes)

    # Finding the size of all the holes in memory.
    diff = np.empty(0)
    i = 1
    while i < len(holes):
        diff = np.append(diff, holes[i] - holes[i - 1] + 1)
        i = i + 2

    # Finding the total amount of memory the process needs.
    process_size = sum(process.address_space)

    # Deciding which hole to use.
    fit_hole = np.where(diff >= process_size)[0]
    diff = [diff[fits] for fits in fit_hole]
    if len(fit_hole) == 0:
        return False
    else:
        if fit_algorithm == FIRST_FIT:
            fit_hole = fit_hole[0]
        elif fit_algorithm == BEST_FIT:
            fit_hole = fit_hole[np.argmin(diff)]
        elif fit_algorithm == WORST_FIT:
            fit_hole = fit_hole[np.argmax(diff)]

    # Using the hole.
    memory[holes[fit_hole * 2]:holes[fit_hole * 2] + process_size] = process.process_id
    process.block_indexes = [holes[fit_hole * 2], holes[fit_hole * 2] + process_size]
    return True


# Running the paging algorithm to place processes in memory.
def pag(process, page_size):
    # Finding the total amount of memory the process needs.
    process_size = sum(process.address_space)

    free_pages = np.where(page_table == 0)[0]

    # Returning false if the system doesn't have enough space to run the process.
    if len(free_pages) * page_size < process_size:
        return False

    # Finding how many pages the process needs.
    page_number = int(process_size / page_size)
    if process_size % page_size != 0:
        page_number = page_number + 1

    # Adding the process to the pages it needs.
    process.pages.clear()
    i, j = 0, 0
    while i < len(page_table) and j < page_number:
        if page_table[i] == 0:
            page_table[i] = process.process_id
            process.pages.append(i)
            j = j + 1
        i = i + 1

    return True


# Running the memory segment algorithm to place processes in memory.
def seg(process, fit_algorithm):
    # Finding the total amount of memory the process needs.
    process_size = sum(process.address_space)

    hole = find_seg_holes()  # Finding all the holes in memory.

    tup_list = []
    for i in range(0, len(process.address_space)):
        address_size = process.address_space[i]

        # Finding the sizes of all the holes.
        sizes = np.empty(0, dtype=int)
        for x in hole:
            sizes = np.append(sizes, int(x[2]))

        # Finding the indexes of all the holes that are big enough.
        good_hole_indexes = np.empty(0, dtype=int)
        for x in range(0, len(sizes)):
            if sizes[x] >= address_size:
                good_hole_indexes = np.append(good_hole_indexes, x)

        # If no hole is big enough return false, we cannot add that process right now.
        if len(good_hole_indexes) == 0:
            return False

        # Determining which hole to use based on the fit algorithm,
        good_holes = hole[good_hole_indexes]
        if fit_algorithm == BEST_FIT:
            hole_to_use = np.argmin(good_holes[:, 2])
            hole_to_use = good_holes[hole_to_use]
        elif fit_algorithm == WORST_FIT:
            hole_to_use = np.argmax(good_holes[:, 2])
            hole_to_use = good_holes[hole_to_use]
        else:
            hole_to_use = good_holes[0]

        # Using the hole.
        tup = (hole_to_use[0], address_size, process.process_id, i)
        tup_list.append(tup)

        # Updating the hole table.
        used_hole = np.where((hole[:, 0] == hole_to_use[0]) & (hole[:, 1] == hole_to_use[1]) & (hole[:, 2] == hole_to_use[2]))[0][0]
        if hole_to_use[2] == address_size:
            hole = np.delete(hole, used_hole, axis=0)
        else:
            hole[used_hole] = (hole[used_hole][0] + address_size, hole[used_hole][1], hole[used_hole][2] - address_size)

    # Adding the segments to the segment table when the process could be fully added to the system.
    for i in tup_list:
        segment_table.append(i)

    return True


def memory_manager(input_queue, running_processes, policy, fit_algorithm, page_size, sim_time, output_file,
                   process_num):
    global memory
    global free_memory
    added = True

    i = 0
    while i < len(input_queue):

        # Doing this because for some reason you can't use globals in a match statement.
        if policy == VSP:
            added = vsp(fit_algorithm, input_queue[i])
        elif policy == PAG:
            added = pag(input_queue[i], page_size)
        elif policy == SEG:
            added = seg(input_queue[i], fit_algorithm)

        if added:
            # Moving the process to the running processes list and print the move out.
            input_queue[i].admitted_time = sim_time
            running_processes.append(input_queue[i])
            output_file.write(s_space + "MM moves Process " + str(input_queue[i].process_id) + " to memory\n")
            print(s_space + "MM moves Process " + str(input_queue[i].process_id) + " to memory")
            input_queue.pop(i)
            output_file.write(s_space + "Input Queue:[" + ' '.join([str(x) for x in input_queue]) + "]\n")
            print(s_space + "Input Queue:[" + ' '.join([str(x) for x in input_queue]) + "]")
            print_mem(output_file, policy, page_size, process_num)
        else:
            i = i + 1

    return


def run_mem_sim(processes, policy, fit_algorithm, page_size, output_file):
    sim_time = 0
    input_queue = []
    running_processes = []
    turnaround = []
    process_num = len(processes)

    while (len(processes) > 0 or len(input_queue) > 0 or len(running_processes) > 0) and sim_time < 100000:
        output_file.write("t = " + str(sim_time) + ":")
        print("t = " + str(sim_time) + ":", end="")
        first_time = True

        # Removing things from the running processes list.
        i = 0
        first_time = True
        while i < len(running_processes):
            x = running_processes[i]
            if x.admitted_time + x.lifetime == sim_time:
                turnaround.append(sim_time - x.arrival_time)
                running_processes.pop(i)
                free_mem(policy, x)

                # Printing out the processes as the complete.
                gap = " " if first_time else s_space
                first_time = False
                output_file.write(gap + "Process " + str(x.process_id) + " completes\n")
                print(gap + "Process " + str(x.process_id) + " completes")
                print_mem(output_file, policy, page_size, process_num)
            else:
                i = i + 1

        # Adding things to the input queue.
        i = 0
        while i < len(processes):
            if sim_time == processes[i].arrival_time:
                input_queue.append(processes[i])

                # Writing the files being added to memory to the file.
                gap = " " if first_time else s_space
                first_time = False
                output_file.write(gap + "Process " + str(processes[i].process_id) + " arrives\n")
                output_file.write(s_space + "Input Queue:[" + ' '.join([str(x) for x in input_queue]) + "]\n")
                print(gap + "Process " + str(processes[i].process_id) + " arrives")
                print(s_space + "Input Queue:[" + ' '.join([str(x) for x in input_queue]) + "]")
                processes.pop(i)
            else:
                i = i + 1
        # Calling the memory manager to add the process to memory.
        memory_manager(input_queue, running_processes, policy, fit_algorithm, page_size, sim_time, output_file,
                       process_num)

        # Scanning forwards to the next time step where something happens.
        smallest = np.inf
        for x in processes:
            if x.arrival_time > sim_time:
                if x.arrival_time < smallest:
                    smallest = x.arrival_time

        for x in running_processes:
            if x.admitted_time + x.lifetime > sim_time:
                if x.admitted_time + x.lifetime < smallest:
                    smallest = x.admitted_time + x.lifetime

        sim_time = smallest
        output_file.write("\n")
        print()

    # Formatting the turnaround time.
    avg = sum(turnaround) / len(turnaround)
    if avg.is_integer():
        avg = "{:.1f}".format(avg)
    else:
        avg = "{:.2f}".format(avg)

    # Writing out the turnaround time.
    output_file.write("Average Turnaround Time: " + str(avg) + "\n")
    print("Average Turnaround Time: " + str(avg))


# Freeing up memory based on the memory management policy.
def free_mem(policy, process):
    if policy == VSP:
        memory[process.block_indexes[0]:process.block_indexes[1]] = 0
    elif policy == SEG:
        i = 0
        while i < len(segment_table):
            if segment_table[i][2] == process.process_id:
                segment_table.pop(i)
            else:
                i = i + 1
    elif policy == PAG:
        for i in process.pages:
            page_table[i] = 0


def print_mem(output_file, policy, page_size, process_num):
    output_file.write(s_space + "Memory Map: \n")
    print(s_space + "Memory Map: ")

    if policy == VSP:
        # Finding the starting and ending indexes of all the holes in memory.
        holes = [0]
        for i in range(1, len(memory) - 1):
            if (memory[i - 1] - memory[i] != 0) or (memory[i] - memory[i + 1] != 0):
                holes.append(i)

        holes.append(len(memory) - 1)

        holes = np.array(holes) # Converting the holes array to a numpy array.

        j = 0
        # Printing out all the spaces in the memory.
        while j < len(holes):
            process_id = "Process " + str(int(memory[holes[j]])) if memory[holes[j]] != 0 else "Hole"
            output_file.write(d_space + str(holes[j]) + "-" + str(holes[j + 1]) + ": " + process_id + "\n")
            print(d_space + str(holes[j]) + "-" + str(holes[j + 1]) + ": " + process_id)
            j = j + 2

    elif policy == SEG:

        hole = find_seg_holes()

        i, j = 0, 0
        while i < len(segment_table) or j < len(hole):

            if j >= len(hole):  # If there are no more holes print out remaining segments.
                process_id = "Process " + str(segment_table[i][2]) + ", Segment " + str(segment_table[i][3])
                index1 = str(segment_table[i][0])
                index2 = str(segment_table[i][0] + segment_table[i][1] - 1)
                i = i + 1
            elif i >= len(segment_table):  # If there is no more segments print out the holes.
                process_id = "Hole"
                index1 = str(hole[j][0])
                index2 = str(hole[j][1] - 1)
                j = j + 1
            elif segment_table[i][0] > hole[j][0]:  # If there is a hole before the next segment print it out.
                process_id = "Hole"
                index1 = str(hole[j][0])
                index2 = str(hole[j][1] - 1)
                j = j + 1
            else:  # Print the segment out.
                process_id = "Process " + str(segment_table[i][2]) + ", Segment " + str(segment_table[i][3])
                index1 = str(segment_table[i][0])
                index2 = str(segment_table[i][0] + segment_table[i][1] - 1)
                i = i + 1

            # Print the values out.
            output_file.write(d_space + index1 + "-" + index2 + ": " + process_id + "\n")
            print(d_space + index1 + "-" + index2 + ": " + process_id)

    elif policy == PAG:
        page_count = np.zeros(process_num)
        i = 0
        while i < len(page_table):  # Printing out the pages.
            if page_table[i] != 0:
                page_count[page_table[i] - 1] = page_count[page_table[i] - 1] + 1
            page_number = str(int(page_count[page_table[i] - 1]))
            j = i + 1
            if page_table[i] != 0:  # If the frame is owned print who owns it.
                process_id = "Process " + str(page_table[i]) + ", Page " + page_number
            else:  # If the frame is free print it as so.
                process_id = "Free Frame(s)"
                while j < len(page_table) and page_table[j] == 0:
                    j = j + 1

            index1 = str(int(i * page_size))
            index2 = str(int(j * page_size - 1))
            output_file.write(d_space + index1 + "-" + index2 + ": " + process_id + "\n")
            print(d_space + index1 + "-" + index2 + ": " + process_id)
            i = j


# Finding all the holes in memory using the segmentation table.
def find_seg_holes():
    # Finding all the holes in memory.
    segment_table.sort(key=lambda x: x[0])
    hole = np.empty((0, 3), dtype=int)
    prev_value = 0
    for i in segment_table:  # For every element in the segment table find if there is a hole before it.
        if i[0] > prev_value:
            hole = np.concatenate((hole, np.array([(prev_value, i[0], i[0] - prev_value)])))
        prev_value = i[0] + i[1]

    if len(segment_table) == 0:  # Add the whole memory as a hole if there is nothing in memory right now.
        hole = np.concatenate((hole, np.array([(0, len(memory), len(memory))])))
        return hole

    # Adding a hole at the end if there is a hole at the end of memory.
    segment = segment_table[len(segment_table) - 1]
    if len(memory) > (segment[0] + segment[1]):
        hole = np.concatenate((hole, np.array([(segment[0] + segment[1], len(memory), len(memory) - (segment[0] + segment[1]))])))

    return hole


if __name__ == '__main__':
    user_input = True
    mem_size = 0
    mem_policy = 0
    page_frame_size = 0
    fit_alg = 0
    read_processes = []

    # Reading in input from the user.
    while user_input:
        try:
            mem_size = int(input("Size of Memory: "))
            if mem_size > 30000:
                print("Error: Memory size too large.")
                continue
            mem_policy = int(input("Management Policy: (1: VSP, 2: PAG, 3: SEG): "))

            if mem_policy != VSP and mem_policy != PAG and mem_policy != SEG:
                print("Error: Management Policy Not Valid.")
                continue

            if mem_policy != PAG:
                fit_alg = int(input("Fit Algorithm (1: First Fit, 2: Best Fit, 3: Worst-Fit): "))
                if fit_alg != FIRST_FIT and fit_alg != BEST_FIT and fit_alg != WORST_FIT:
                    print("Error: Algorithm selected not valid.")
                    continue
            else:
                page_frame_size = int(input("Page/Frame Size: "))
                page_num = int(mem_size / page_frame_size)
                if mem_size % page_frame_size != 0:
                    page_num = page_num + 1
                page_table = np.zeros(page_num, dtype=int)

            user_input = False

        except ValueError:
            print("Error: Input not an integer.")
            continue

    memory = np.zeros(mem_size)

    read_processes = read_file()

    # Naming the output file based on the algorithm used.
    if fit_alg == FIRST_FIT:
        fit_str = "first"
    elif fit_alg == BEST_FIT:
        fit_str = "best"
    else:
        fit_str = "worst"

    file_name = ""
    if mem_policy == VSP:
        file_name = "vsp_" + fit_str
    elif mem_policy == SEG:
        file_name = "seg-" + fit_str
    elif mem_policy == PAG:
        file_name = "P" + str(page_frame_size)

    output = open(file_name + ".out", "w+")  # Opening the output file.

    # Running the memory simulator.
    run_mem_sim(read_processes, mem_policy, fit_alg, page_frame_size, output)
