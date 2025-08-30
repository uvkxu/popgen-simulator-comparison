import os

def initial_population_activity(config):
    pop_names = [pop["name"] for pop in config.get("populations", [])]
    activity = {pid: True for pid in pop_names}  # Default: aktiv
    
    for event in config.get("demography_events", []):
        type = event["type"]
        
        if type == "population_split":
            activity[event["ancestral"]] = False
        
        elif type == "population_admixture":
            for a in event["ancestral"]:
                activity[a] = False
    
    return activity

def get_populations(config):
    pops = config.get("populations", [])

    ancestral_names = set()
    for event in config.get("demography_events", []):
        if event["type"] == "population_admixture":
            ancestral_names.update(event.get("ancestral", []))

    pops = [pop for pop in pops if pop.get("name") not in ancestral_names]

    return pops

def check_events_sorted(config):
    events = config.get("demography_events", [])
    times = [event["time"] for event in events]
    
    return times == sorted(times)

