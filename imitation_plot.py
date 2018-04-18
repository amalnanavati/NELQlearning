import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors=["blue","red","green","yellow", "black","cyan","magenta"]

def plot(plot_agent):

    fig = plt.figure()

    chart_1 = fig.add_subplot(211)
    chart_1.set_title('Average Jellybeans Collected Per Timestep (y) vs. Timesteps (x)')
    chart_1.grid(True)

    chart_2 = fig.add_subplot(212)
    chart_2.set_title('Average Jellybeans Collected Per Clocktime (y) vs. Clocktime (x)')
    chart_2.grid(True)


    for i in range(len(plot_agent)):

        #plot_data gets information about agents != 0
        plot_data = plot_agent[i]

        #base_data gets data of agent 0
        base_data = plot_agent[0]

        #shift step is the step this agent spawned
        shift_step = plot_data[0][0] - base_data[0][0]

        #shift clock is the clock this agent spawned
        shift_clock = plot_data[1][0] - base_data[0][0]

        #do the shift
        plot_data_cumu_rew_over_world_time = []
        plot_data_cumu_rew_over_wall_time = []
        for k in range(len(plot_data[0])):
            plot_data[0][k] -= shift_step
            # plot_data[1][k] -= shift_clock
            plot_data_cumu_rew_over_world_time.append(plot_data[2][k]/plot_data[0][k])
            plot_data_cumu_rew_over_wall_time.append(plot_data[2][k]/plot_data[1][k])

        # Graph cumulative reward (y) versus time (x)
        # l1, = chart_1.plot(plot_data[0],plot_data[2],color=colors[i],label="agent_"+str(i))
        # l2, = chart_2.plot(plot_data[1],plot_data[2],color=colors[i],label="agent_"+str(i))

        # Graph Cumulative Reward divided by time (y) versus time (x)
        # length = 100000
        subsetSize = len(plot_data[0])#length/200#
        l1, = chart_1.plot(plot_data[0][:subsetSize],plot_data_cumu_rew_over_world_time[:subsetSize],color=colors[i],label="agent_"+str(i))
        l2, = chart_2.plot(plot_data[1][:subsetSize],plot_data_cumu_rew_over_wall_time[:subsetSize],color=colors[i],label="agent_"+str(i))

        # Graph Cumulative Reward over last 50000 timesteps divided by time in last 50000 timesteps (y) versus time (x)
        # lastTimesteps = 50000
        # interval = lastTimesteps/200
        # plot_data_derivative_world_time = [float(plot_data[2][k] - plot_data[2][k-interval])/(plot_data[0][k] - plot_data[0][k-interval]) for k in xrange(interval, len(plot_data[2]))]
        # plot_data_derivative_clock_time = [float(plot_data[2][k] - plot_data[2][k-interval])/(plot_data[1][k] - plot_data[1][k-interval]) for k in xrange(interval, len(plot_data[2]))]
        #
        # l1, = chart_1.plot(plot_data[0][0:len(plot_data[0])-interval],plot_data_derivative_world_time,color=colors[i],label="agent_"+str(i))
        # l2, = chart_2.plot(plot_data[1][0:len(plot_data[1])-interval],plot_data_derivative_clock_time,color=colors[i],label="agent_"+str(i))


    # label_chart = fig.add_subplot(222)
    #
    # labels = ['agent_'+str(i) for i in xrange(len(plot_agent))]
    # xcolor = colors[0:len(plot_agent)]
    #
    # patches = [
    #     mpatches.Patch(color=color, label=label) for label, color in zip(labels, xcolor)
    #     ]
    # label_chart.legend(patches, labels, loc='center', frameon=False)

    fig.tight_layout()

    plt.show()

def plot_compare(plot_agent_1,plot_agent_2):
    fig = plt.figure()

    chart_1 = fig.add_subplot(211)
    chart_1.set_title('acc. rewards per timestep')
    chart_1.grid(True)

    chart_2 = fig.add_subplot(212)
    chart_2.set_title('acc. rewards per time(clock)')
    chart_2.grid(True)

    fig.tight_layout()

    for i in range(len(plot_agent_1)):
        plot_data_1 = plot_agent_1[i]
        plot_data_2 = plot_agent_2[i]

        diff_rew = list()

        for k in range(len(plot_data_1[2])):
            diff_rew.append(plot_data_1[2][k] - plot_data_2[2][k])

        chart_1.plot(plot_data_1[0],diff_rew,color=colors[i])
        chart_2.plot(plot_data_1[1],diff_rew,color=colors[i])

    plt.show()

def load(out_dir,num_agents):
    plot_agent = list()

    for i in range(num_agents):

        f =  open(out_dir+'/plot_'+str(i)+'.txt')


        plot_data = list()
        steps = list()
        clock = list()
        rew = list()
        loss = list()

        for line in f:
            plot_line = line.replace('\n','').split(" ")

            steps.append(int(plot_line[0]))
            clock.append(float(plot_line[1]))
            rew.append(float(plot_line[2]))
            loss.append(float(plot_line[3]))

            plot_data.append(steps)
            plot_data.append(clock)
            plot_data.append(rew)
            plot_data.append(loss)

        plot_agent.append(plot_data)

    return plot_agent

def main():

    num_agents = 5

    plot_agent = load('outputs',num_agents)

    plot(plot_agent)

    #to use plot compare, you have to save the runs in
    #different directories, then load both and call compare

    # plot_compare(plot_agent,plot_agent)

if __name__ == '__main__':
    main()
