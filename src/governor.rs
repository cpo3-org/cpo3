//! Adaptive centering schedule, using cpo3's native stopping score.

use crate::{Settings, mehrotra::StepKind};

const HISTORY: usize = 9;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Observation {
    pub(crate) mu_step: f64,
    pub(crate) delta_path: f64,
    pub(crate) stopping_score: f64,
}

pub(crate) struct Governor {
    max_recenters: usize,
    sigma_floor: f64,
    history: [Observation; HISTORY],
    len: usize,
    previous_spike: Option<f64>,
    pending: bool,
    pacing: bool,
    recenter_steps: usize,
}

#[cfg(test)]
impl Default for Governor {
    fn default() -> Self {
        Self::new(&Settings::default())
    }
}

impl Governor {
    pub(crate) fn new(settings: &Settings) -> Self {
        Self {
            max_recenters: settings.max_recenters,
            sigma_floor: settings.recenter_sigma_floor,
            history: [Observation::default(); HISTORY],
            len: 0,
            previous_spike: None,
            pending: false,
            pacing: false,
            recenter_steps: 0,
        }
    }

    pub(crate) fn recenter_steps(&self) -> usize {
        self.recenter_steps
    }

    /// Once the recenter budget is exhausted, observations cannot change
    /// the schedule. `next_step` must still apply the persistent sigma floor.
    pub(crate) fn needs_observation(&self) -> bool {
        self.recenter_steps < self.max_recenters
    }

    /// Consume a scheduled recenter only when the next step actually starts.
    pub(crate) fn next_step(&mut self) -> StepKind {
        if std::mem::take(&mut self.pending) {
            self.recenter_steps += 1;
            StepKind::Recenter
        } else {
            StepKind::Ordinary {
                sigma_floor: if self.pacing { self.sigma_floor } else { 0.0 },
            }
        }
    }

    /// Observe accepted, nonterminal states, before their next normalization.
    pub(crate) fn observe(&mut self, now: Observation) {
        if self.len == HISTORY {
            self.history.rotate_left(1);
        } else {
            self.len += 1;
        }
        self.history[self.len - 1] = now;
        let mut fire =
            self.len == HISTORY && self.history[0].stopping_score < 1.2 * now.stopping_score;
        if !fire && self.len >= 3 {
            let old = self.history[self.len - 3];
            if now.mu_step < old.mu_step && now.delta_path > 20.0 * old.delta_path {
                fire = self
                    .previous_spike
                    .is_some_and(|score| now.stopping_score > score / 5.0);
                self.previous_spike = Some(now.stopping_score);
            }
        }
        if fire && self.recenter_steps < self.max_recenters {
            self.pending = true;
            self.pacing = true;
            self.len = 0;
            self.previous_spike = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(mu_step: f64, delta_path: f64, stopping_score: f64) -> Observation {
        Observation {
            mu_step,
            delta_path,
            stopping_score,
        }
    }

    #[test]
    fn stagnation_schedules_next_step_and_pacing_survives_the_eight_recenter_cap() {
        let mut governor = Governor::default();
        for count in 0..8 {
            for _ in 0..8 {
                governor.observe(obs(1.0, 1.0, 10.0));
            }
            assert!(!governor.pending);
            governor.observe(obs(1.0, 1.0, 10.0));
            assert_eq!(governor.recenter_steps(), count);
            assert_eq!(governor.len, 0);
            assert_eq!(governor.next_step(), StepKind::Recenter);
            assert_eq!(governor.recenter_steps(), count + 1);
            assert_eq!(
                governor.next_step(),
                StepKind::Ordinary {
                    sigma_floor: Settings::default().recenter_sigma_floor
                }
            );
        }
        for _ in 0..20 {
            governor.observe(obs(1.0, 1.0, 10.0));
        }
        assert!(!governor.pending);
        assert_eq!(governor.recenter_steps(), 8);
        assert_eq!(
            governor.next_step(),
            StepKind::Ordinary {
                sigma_floor: Settings::default().recenter_sigma_floor
            }
        );
    }

    #[test]
    fn strict_spike_and_progress_thresholds_follow_the_reference() {
        for (last_score, fires) in [(20.0, false), (20.1, true)] {
            let mut governor = Governor::default();
            // First spike is remembered, never triggers by itself.
            governor.observe(obs(10.0, 1.0, 200.0));
            governor.observe(obs(9.0, 1.0, 150.0));
            governor.observe(obs(8.0, 21.0, 100.0));
            assert_eq!(governor.previous_spike, Some(100.0));
            assert!(!governor.pending);
            governor.observe(obs(7.0, 20.0, 80.0)); // Exactly 20x is not a spike.
            assert_eq!(governor.previous_spike, Some(100.0));
            governor.observe(obs(6.0, 421.0, last_score));
            assert_eq!(governor.pending, fires);
            assert_eq!(
                governor.previous_spike,
                if fires { None } else { Some(last_score) }
            );
        }
        let mut governor = Governor::default();
        governor.observe(obs(1.0, 1.0, 12.0));
        for _ in 0..8 {
            governor.observe(obs(1.0, 1.0, 10.0));
        }
        assert!(!governor.pending); // Equality at 1.2x does not fire.
        governor.observe(obs(1.0, 1.0, 10.0));
        assert!(governor.pending); // The rolling window dropped the first point.
    }

    #[test]
    fn stagnation_takes_priority_over_a_simultaneous_spike_at_the_cap() {
        let mut governor = Governor {
            recenter_steps: Settings::default().max_recenters,
            previous_spike: Some(123.0),
            ..Governor::default()
        };
        for _ in 0..8 {
            governor.observe(obs(2.0, 1.0, 10.0));
        }
        governor.observe(obs(1.0, 100.0, 10.0));
        // The stagnation branch suppresses spike bookkeeping even when the
        // cap prevents scheduling another recenter.
        assert_eq!(governor.previous_spike, Some(123.0));
        assert!(!governor.pending);
    }

    #[test]
    fn nondecreasing_mu_cannot_trigger_a_spike() {
        let mut governor = Governor::default();
        for delta in [1.0, 1.0, 100.0, 100.0, 10000.0] {
            governor.observe(obs(1.0, delta, 10.0));
        }
        assert_eq!(governor.previous_spike, None);
        assert!(!governor.pending);
    }
}
