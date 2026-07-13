# Artcraft — guide de contribution

Minecraft-like écrit **from scratch** en Rust + wgpu, dans un but d'apprentissage.
Le propriétaire du projet ne connaît pas Rust : la lisibilité prime sur la concision.

## Commandes

```
cargo run          # lance le jeu (fenêtre)
cargo test         # tests unitaires (logique pure : physique, raycast, monde)
cargo fmt          # formatage — à lancer avant chaque commit
cargo clippy       # lints — corriger les avertissements introduits
```

## Philosophie

- **From scratch** : pas de crate toute faite pour ce qui est le cœur du sujet
  (bruit, meshing, physique, éclairage). Les seules dépendances acceptées sont
  l'infrastructure : `wgpu`, `winit`, `glam`, `bytemuck`, `pollster`.
  Demander avant d'ajouter une dépendance.
- **Chaque jalon reste jouable** : ne jamais laisser `main` dans un état où
  `cargo run` ne donne pas un jeu fonctionnel.
- La feuille de route vit dans le `README` — la tenir à jour.

## Architecture (un module = une responsabilité)

- `main.rs` — boucle d'événements winit, routage clavier/souris. Rien d'autre.
- `state.rs` — tout ce qui touche au GPU : pipelines, buffers, render pass.
- `world.rs` — le monde : cycle de vie des chunks, raycast, modification de blocs.
- `chunk.rs` — données d'un chunk, génération du terrain, construction du mesh.
- `player.rs` — physique (AABB, gravité, collisions). Ne connaît pas le GPU.
- `camera.rs` — projection et état des touches. `noise.rs` — bruit de Perlin.
- `texture.rs` — atlas procédural. `*.wgsl` — shaders.

La séparation clé : **la logique de jeu (world, chunk, player, noise) ne doit
jamais dépendre de wgpu/winit**, à l'exception des buffers GPU des meshes.
C'est ce qui permet de la tester avec `cargo test` sans fenêtre ni GPU.

## Conventions de code

- Commentaires en français, qui expliquent le **pourquoi** (contraintes,
  pièges, invariants), pas le quoi. Chaque module commence par une ou deux
  phrases résumant son rôle.
- Doc-comments (`///`) sur les types et fonctions publics.
- Pas de `unwrap()`/`expect()` dans la boucle de jeu ; réservés à
  l'initialisation, où planter tôt est le bon comportement.
- Les constantes de gameplay (vitesses, portées, budgets) sont des `const`
  nommées en tête de module, jamais des nombres magiques en plein code.
- Coordonnées : `IVec3` = coordonnées de bloc (monde), `(i32, i32)` =
  coordonnées de chunk, `Vec3` = position continue. Ne pas les mélanger.
- Toute logique non triviale (collision, raycast, éclairage…) a un test
  unitaire qui raconte un scénario concret.

## Workflow

1. Développer par jalon ; à la fin : `cargo fmt`, `cargo test`, `cargo run`
   avec vérification visuelle (capture d'écran).
2. Un commit par jalon fonctionnel, message en français décrivant le contenu.
3. En cas de doute sur une orientation gameplay/technique : demander au
   propriétaire plutôt que trancher seul.
