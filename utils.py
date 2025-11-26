# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 02:00:47 2025

@author: SHAIK RIFSHU
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def generate_gantt_chart(schedule_log: List[Dict[str, Any]], scheduler_type: str):
    """ Generates and saves a Gantt chart using matplotlib. 
    
    Args:
        schedule_log (list): A list of dictionaries, where each dict represents
                             a completed or preempted operation.
        scheduler_type (str): The name of the scheduler (e.g., 'FIFO', 'ACO') for the title.
    """
    
    if not schedule_log:
        print("No schedule data available to plot Gantt chart.")
        return

    df = pd.DataFrame(schedule_log)
    
    # 1. Prepare data for plotting
    jobs = df['Job'].unique()
    machines = df['Machine'].unique()
    machines.sort() 

    # 2. Assign unique color to each Job
    try:
        # Using a distinct colormap
        colors = plt.cm.get_cmap('tab20', len(jobs))
    except ValueError:
        colors = plt.cm.get_cmap('hsv', len(jobs))

    job_color_map = {job: colors(i) for i, job in enumerate(jobs)}

    fig, ax = plt.subplots(figsize=(14, 7))

    # 3. Plotting logic
    for i, machine in enumerate(machines):
        df_machine = df[df['Machine'] == machine]
        
        # Plot bars for operations on this machine
        for index, row in df_machine.iterrows():
            job_id = row['Job']
            start = row['Start']
            duration = row['Finish'] - start
            
            # Plot the bar for the operation
            ax.barh(
                machine, 
                duration, 
                left=start, 
                color=job_color_map[job_id], 
                edgecolor='black',
                linewidth=0.5
            )

    # 4. Final Aesthetics
    ax.set_title(f'Gantt Chart for {scheduler_type.upper()} Scheduler', fontsize=16)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.invert_yaxis() 
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Create Legend
    legend_handles = [plt.Rectangle((0,0),1,1, fc=job_color_map[job]) for job in jobs]
    legend_labels = [f'Job {j}' for j in jobs]
    
    # Place legend outside the plot area
    ax.legend(
        legend_handles, 
        legend_labels, 
        title="Jobs", 
        loc='upper left', 
        bbox_to_anchor=(1.01, 1), 
        ncol=1, 
        fontsize=8, 
        fancybox=True, 
        shadow=True
    )

    plt.subplots_adjust(right=0.75) 

    # Save the figure
    plot_file = f"gantt_chart_{scheduler_type.lower()}.png"
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"\n🖼️ Gantt Chart saved: {plot_file}")
    plt.close(fig)