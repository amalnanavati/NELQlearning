import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors=["blue","red","green","yellow", "black","cyan","magenta"]

def plot(plot_agent):

    fig = plt.figure()

    chart_1 = fig.add_subplot(211)
    chart_1.set_title('Average Jellybeans Collected Per Timestep  vs. Timesteps with 500K steps')
    chart_1.grid(True)
    chart_1.set_xlabel('timestep')
    chart_1.set_ylabel('average jellybeans collected\nper timestep', multialignment='center')

    chart_2 = fig.add_subplot(212)
    chart_2.set_title('Average Jellybeans Collected Per Timestep vs. Clocktime with 100K steps')
    chart_2.grid(True)
    chart_2.set_xlabel('timestep')
    chart_2.set_ylabel('average jellybeans collected\nper timestep')
    chart_2.set_xlim(0,100000)


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
        for k in range(len(plot_data[0])):
            plot_data[0][k] -= shift_step
            plot_data_cumu_rew_over_world_time.append(plot_data[2][k]/plot_data[0][k])


        l1, = chart_1.plot(plot_data[0],plot_data_cumu_rew_over_world_time,color=colors[i],label="agent_"+str(i))
        l2, = chart_2.plot(plot_data[0],plot_data_cumu_rew_over_world_time,color=colors[i],label="agent_"+str(i))


    fig.tight_layout()

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

def load_avg(out_dir_list,num_agents):

    plot_agent = list()

    for i in range(num_agents):

        plot_files = list()
    
        for out_dir in out_dir_list:
            with open(out_dir+'/plot_'+str(i)+'.txt') as f:
                lines = f.readlines()
            lines = [x.strip() for x in lines]

            plot_files.append(lines)

        plot_data = list()

        steps = list()
        clock = list()
        rew = list()
        loss = list()

        for l in range(len(plot_files[0])): #assume all files have same n of lines

            sum_steps = 0
            sum_clock = 0
            sum_rew = 0
            sum_loss = 0
           
            for k in range(len(plot_files)):
                line_split = plot_files[k][l].replace('\n','').split(" ")

                sum_steps += int(line_split[0])
                sum_clock += float(line_split[1])
                sum_rew += float(line_split[2])
                sum_loss += float(line_split[3])

            steps.append(sum_steps/len(out_dir_list))
            clock.append(sum_clock/len(out_dir_list))
            rew.append(sum_rew/len(out_dir_list))
            loss.append(sum_loss/len(out_dir_list))

        plot_data.append(steps)
        plot_data.append(clock)
        plot_data.append(rew)
        plot_data.append(loss)

        plot_agent.append(plot_data)

    return plot_agent

def main():

    plot_agent = load_avg(['res/out0','res/out1','res/out2','res/out3','res/out4','res/out5','res/out6','res/out7','res/out8'],5)

    plot(plot_agent)

if __name__ == '__main__':
    main()
