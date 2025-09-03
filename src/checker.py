import os

def initial_population_activity(config):
    pop_names = [pop["name"] for pop in config.get("populations", [])]
    activity = {pid: True for pid in pop_names}  # Default: aktiv
    
    for event in config.get("demography_events", []):
        type = event["type"]
        
        if type == "population_split":
            activity[event["ancestral"]] = False
    
    return activity

def check_events_sorted(config):
    events = config.get("demography_events", [])
    times = [event["time"] for event in events]
    
    return times == sorted(times)

