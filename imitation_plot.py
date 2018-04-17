import matplotlib.pyplot as plt

def plot(plot_agent):

    fig = plt.figure()
    colors=["blue","red","green","yellow"]

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
        for k in range(len(plot_data[0])):
            plot_data[0][k] -= shift_step
            plot_data[1][k] -= shift_clock

        chart_1.plot(plot_data[0],plot_data[2],color=colors[i])
        chart_2.plot(plot_data[1],plot_data[2],color=colors[i])

    plt.show()

def plot_compare(plot_agent):
    fig = plt.figure()
    colors=["blue","red","green","yellow"]

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

    num_agents = 2

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
    plot_compare(plot_agent)

if __name__ == '__main__':
    main()
