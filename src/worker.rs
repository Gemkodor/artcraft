//! Pool de threads pour la génération et le meshing des chunks.
//!
//! Le thread principal envoie des jobs dans un canal et récupère les
//! résultats de manière non bloquante à chaque frame : les gros calculs
//! (bruit de Perlin, construction de mesh) ne le touchent jamais, ce qui
//! élimine les micro-saccades. Les threads se terminent d'eux-mêmes quand le
//! pool est détruit (les canaux se ferment).

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::chunk::{self, Chunk, ChunkNeighbors};
use crate::mesh::Vertex;
use crate::noise::Noise;

pub enum Job {
    /// Générer les données de blocs du chunk (bruit de Perlin, grottes).
    Generate { coord: (i32, i32) },
    /// Construire le mesh d'un chunk dont on fournit les données.
    /// `version` permet au thread principal d'écarter un résultat périmé si
    /// un bloc a été modifié entre l'envoi du job et son retour.
    Mesh {
        coord: (i32, i32),
        version: u64,
        center: Arc<Chunk>,
        neighbors: ChunkNeighbors,
    },
}

pub enum JobResult {
    Generated {
        coord: (i32, i32),
        chunk: Chunk,
    },
    Meshed {
        coord: (i32, i32),
        version: u64,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    },
}

pub struct WorkerPool {
    job_tx: Sender<Job>,
    result_rx: Receiver<JobResult>,
}

impl WorkerPool {
    pub fn new(noise: Noise) -> Self {
        let (job_tx, job_rx) = channel::<Job>();
        let (result_tx, result_rx) = channel::<JobResult>();
        // Le Receiver de mpsc n'est pas partageable tel quel : on le protège
        // par un Mutex pour que chaque worker puisse y piocher un job.
        let job_rx = Arc::new(Mutex::new(job_rx));

        // On laisse deux cœurs au thread principal et au pilote graphique.
        let workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .saturating_sub(2)
            .clamp(1, 4);

        for id in 0..workers {
            let job_rx = Arc::clone(&job_rx);
            let result_tx = result_tx.clone();
            thread::Builder::new()
                .name(format!("chunk-worker-{id}"))
                .spawn(move || {
                    loop {
                        // Le lock ne couvre que la prise du job, pas son
                        // exécution, sinon les workers se sérialiseraient.
                        let job = match job_rx.lock() {
                            Ok(rx) => rx.recv(),
                            Err(_) => return,
                        };
                        let Ok(job) = job else {
                            return; // canal fermé : le monde a été détruit
                        };
                        let result = match job {
                            Job::Generate { coord } => JobResult::Generated {
                                coord,
                                chunk: Chunk::generate(&noise, coord.0, coord.1),
                            },
                            Job::Mesh {
                                coord,
                                version,
                                center,
                                neighbors,
                            } => {
                                let (vertices, indices) =
                                    chunk::build_mesh(&center, &neighbors, coord.0, coord.1);
                                JobResult::Meshed {
                                    coord,
                                    version,
                                    vertices,
                                    indices,
                                }
                            }
                        };
                        if result_tx.send(result).is_err() {
                            return;
                        }
                    }
                })
                .expect("échec de création d'un thread worker");
        }

        Self { job_tx, result_rx }
    }

    pub fn submit(&self, job: Job) {
        // Un échec d'envoi signifie que tous les workers sont morts ; le jeu
        // continue alors sans nouveaux chunks plutôt que de planter.
        let _ = self.job_tx.send(job);
    }

    /// Tous les résultats déjà prêts, sans jamais bloquer.
    pub fn drain_results(&self) -> Vec<JobResult> {
        self.result_rx.try_iter().collect()
    }
}
