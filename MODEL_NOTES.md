# EventFlow Mapping Notes

This initial simulation can be translated into an EventFlow-like model by treating the intersection as a set of stateful entities driven by timestamped events.

## Entities

- `IntersectionController`
  - state: `current_phase`, `phase_index`
- `Lane[north|south|east|west]`
  - state: `queue_length`, `arrivals`, `departures`, `max_queue`
- `Vehicle`
  - optional future state: `arrival_time`, `wait_time`, `movement`

## Event types

- `phase_change`
  - emitted when the controller advances to a new phase
  - may schedule new `vehicle_departure` opportunities for compatible lanes
- `vehicle_arrival`
  - increments a lane queue
  - may trigger a departure schedule if the lane is green and idle
- `vehicle_departure`
  - decrements a lane queue
  - updates wait-time metrics
  - may schedule the next departure on the same lane if demand remains

## Transition ideas

- `phase_change(NS_GREEN)` enables service on `north` and `south`
- `phase_change(EW_GREEN)` enables service on `east` and `west`
- `vehicle_arrival` mutates queue state and conditionally creates service demand
- `vehicle_departure` consumes queued demand and conditionally chains the next service event

## Why the CSV trace matters

`output/events.csv` is already a useful event log for reconstructing:

- causal ordering
- controller state at each event
- queue lengths after each mutation

That gives us a clean bridge from this imperative simulation to an explicit event/state graph in the next iteration.
