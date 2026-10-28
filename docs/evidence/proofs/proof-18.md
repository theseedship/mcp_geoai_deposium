# Technical evidence 18

Source: `docs/design/geoai-guided-analysis-spec-2026-10-27.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `1c1c2e17c9a94f52a78fc2a3a70e7b080926ac4d4afc257f72855d551ce6ed2e`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

| Lot | Livraison | État vérifié |
|---|---|---|
| P0 — formulaires guidés | Solid [internal record] | Fusionné dans `main` (`1b08406b40b1337d781994e7a4bff31e4fd37f33`), 39 contrôles verts. Résumé replié, options avancées, temporalité et récapitulatif ; brouillons et documents ciblés conservés. |
| P1a — exemple B6 conservé | MCPs [internal record], puis Solid [internal record] | Fusionné et déployé sur le staging the qualified staging environment : reçu typé signé, relecture canonique, facteurs et 149 traces conservées. the contributor/maintainer a confirmé l'affichage le 28 octobre ; cela ne constitue pas une nouvelle qualification scientifique. |
| P1b — résultat NDVI riche | MCPs [internal record], puis Solid [internal record] | Implémentation versionnée et testée : acquisitions conservées, tableau par pages de 25 lignes et graphique de points après relecture autorisée. Les PR liées portent l'état de livraison distant ; aucune courbe n'est reconstruite depuis le Markdown. |
| P2 — dessin de bbox | Extension facultative de la carte | À décider ensuite ; les quatre champs explicites restent le parcours disponible. |

P1b utilise la même conservation, avec un domaine de signature NDVI distinct
et sans grant propre à l'exemple historique. Le formulaire NDVI explicite
fournit les paramètres ; le résultat typé garde les dates d'acquisition exactes,
les zéros, valeurs absentes, doublons et critères réellement retournés.
Les points ne sont pas reliés et aucune journée intermédiaire n'est créée.
Les envois NDVI en langage naturel et les anciens messages textuels gardent
leur parcours existant. Ce lot n'ajoute ni fournisseur, ni calcul scientifique,
ni migration, ni dépendance ; une absence de source ou de licence reste inconnue.

Validation locale P1a : 175 tests MCPs sur `e4de21748d2ff897339ab2acbad71a758a299876`,
458 tests Solid sur `24d46080d3d896a35b2773c81301292fed12adb5`, TypeScript et essais
Chromium du panneau et de la sélection à 375/1280 px. Les tests utilisent des fixtures : ils ne
prouvent ni déploiement, ni nouvelle qualification scientifique. Les limites
B6 restent visibles : adoption scientifique en attente, confiance calibrée
non disponible (`null`) et activation opérationnelle désactivée.

Pour P1b, les suites vérifient conservation/rechargement, profils vides,
révocation des droits et absence d'acquisition à la relecture. Le tableau et
le graphique ont été testés avec le moteur MCPUI réel à 375/1280 px, dont
la pagination locale et la navigation au clavier. Ces fixtures synthétiques
ne prouvent pas une nouvelle acquisition satellite connectée. La vue B6 reçoit
aussi une explication courte de son rôle : classer et expliquer un rapprochement
historique, sans démontrer une future hausse du NDVI.

| Élément | Acquis dans les sources auditées | Évolution proposée |
|---|---|---|
| Entrée GeoAI | Section repliable dans Filtres, visible avec le plugin actif | Résumé compact du mode, du lieu et de la période |
| Type d'analyse | Modes météo et NDVI, brouillons distincts | Libellés guidés ; exemple historique clairement séparé |
| Point météo | GPS, carte facultative, lieux de référence | Réutiliser ces contrôles sans reconstruire de sélecteur |
| Zone NDVI | Bbox WGS84 explicite, quatre champs | Dessin de rectangle facultatif dans un lot ultérieur |
| Temporalité | Météo 1–16 jours ; NDVI période inclusive ≤366 jours | Expliquer le sens de chaque temporalité |
| Résultat météo | Artefact conservé et projection en tableau | Hiérarchie de lecture et couverture effectivement servie |
| Lecteur scientifique | Table paginée et artefact signé | Réutiliser résumé, continuation et Sources / méthode |
| Résultat NDVI interactif | Texte factuel issu du calcul, sans artefact dédié | Contrat typé et conservation avant tableau/courbe riches |
| B6 historique | Cas fixe autorisé, réponse textuelle sans artefact navigateur | Contrat typé et conservation avant panneau riche |

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
