import matplotlib.pyplot as plt

colors=["blue","red","green","yellow", "black"]

def plot(plot_agent):

    fig = plt.figure()

    chart_1 = fig.add_subplot(211)
    chart_1.set_title('acc rewards per timestep')
    chart_1.grid(True)

    chart_2 = fig.add_subplot(212)
    chart_2.set_title('acc rewards per time(clock)')
    chart_2.grid(True)

    fig.tight_layout()

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
            plot_data[1][k] -= shift_clock
            plot_data_cumu_rew_over_world_time.append(plot_data[2][k]/plot_data[0][k])
            plot_data_cumu_rew_over_wall_time.append(plot_data[2][k]/plot_data[1][k])

        # Graph cumulative reward (y) versus time (x)
        # chart_1.plot(plot_data[0],plot_data[2],color=colors[i])
        # chart_2.plot(plot_data[1],plot_data[2],color=colors[i])

        # Graph Cumulative Reward divided by time (y) versus time (x)
        # chart_1.plot(plot_data[0],plot_data_cumu_rew_over_world_time,color=colors[i])
        # chart_2.plot(plot_data[1],plot_data_cumu_rew_over_wall_time,color=colors[i])

        # Computer Derivatives of Reward w.r.t Time
        interval = 50000/200
        plot_data_derivative_world_time = [float(plot_data[2][k] - plot_data[2][k-interval])/(plot_data[0][k] - plot_data[0][k-interval]) for k in xrange(interval, len(plot_data[2]))]
        plot_data_derivative_clock_time = [float(plot_data[2][k] - plot_data[2][k-interval])/(plot_data[1][k] - plot_data[1][k-interval]) for k in xrange(interval, len(plot_data[2]))]

        chart_1.plot(plot_data[0][0:len(plot_data[0])-interval],plot_data_derivative_world_time,color=colors[i])
        chart_2.plot(plot_data[1][0:len(plot_data[1])-interval],plot_data_derivative_clock_time,color=colors[i])

    plt.show()

def plot_compare(plot_agent):
    fig = plt.figure()

    chart_1 = fig.add_subplot(211)
    chart_1.set_title('acc. rewards per timestep')
    chart_1.grid(True)

    chart_2 = fig.add_subplot(212)
    chart_2.set_title('acc. rewards per time(clock)')
    chart_2.grid(True)

    fig.tight_layout()

    for i in range(len(plot_agent)):
        base_data = plot_agent[0]
        plot_data = plot_agent[i]

        if i != 0:

            for k in range(len(plot_data[0])):
                plot_data[2][k] = plot_data[2][k] - base_data[2][k]

        chart_1.plot(plot_data[0],plot_data[2],color=colors[i])
        chart_2.plot(plot_data[1],plot_data[2],color=colors[i])

    plt.show()

def main():

    num_agents = 5

    plot_data = list()
    plot_agent = list()

    for i in range(num_agents):

        f =  open('outputs/plot_'+str(i)+'.txt')

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

    plot(plot_agent)
    # plot_compare(plot_agent)

if __name__ == '__main__':
    main()
