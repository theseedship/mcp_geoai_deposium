# FIERCE — conducteur de démonstration technique dans DEPOSIUM

Préparé le **31 octobre 2026**. Public : évaluateurs techniques et scientifiques.
Périmètre : **capacités géospatiales, intégration, résultats et limites**.

Ce conducteur accompagne l’[annexe technique et ses preuves](fierce-kpi-evidence.en.md).
Il prépare une nouvelle captation ; il ne constitue ni une vidéo déjà réalisée,
ni une nouvelle qualification de l’environnement utilisé. Les preuves citées
conservent leurs dates et leur périmètre d’origine.

## 1. Le fil à raconter

> Nous rendons des données météo et satellitaires accessibles dans un espace de
> travail, avec leurs dates, unités et sources. Nous montrons trois indicateurs
> sur trois pilotes, des analyses conservées et des règles de surveillance.
> Nous avons aussi testé la découverte de motifs : les résultats scientifiques,
> y compris négatifs, restent distincts du fonctionnement de l’interface.

Trois familles de données doivent rester reconnaissables :

| Famille | Ce que l’on montre | Ce que cela représente |
| --- | --- | --- |
| Historique joint N1 | NDVI, température et précipitations, trois pilotes | Acquisitions Sentinel-2 rapprochées de jours ERA5 ; observations espacées, pas une prévision. |
| Prévision actuelle | Trois prochains jours à un point explicite | Sortie d’un fournisseur météo, avec une heure de collecte ; pas une mesure en station. |
| Exemple historique B6 | Bibliothèque de motifs et prévision GFS archivée | Calcul explicatif traçable sur un cas fermé ; pas une probabilité d’événement. |

**Pour le dépôt du livrable, filmer sur `deposium.ai`** : c’est la plateforme SaaS
spécifiée dans le dossier technique. Une répétition sur une autre instance est
utile, mais doit porter son vrai nom d’environnement. Elle ne vaut pas, à elle
seule, captation sur `.ai`.

Les messages ci-dessous sont des **commandes de l’application DEPOSIUM**. Les
commandes `pilot_data_read`, `geoai_weather` ou `level2_historical_case` ne sont
pas des noms d’outils natifs à envoyer tels quels au serveur Python indépendant.
Le [guide des 16 outils MCP](../../reference/mcp-tools.md) décrit ce second
niveau ; l’[architecture](../architecture.md) distingue serveur et application.

## 2. Préparer une répétition qui corresponde au film

Avant d’enregistrer :

1. Dans l’environnement choisi, ouvrir un espace de démonstration avec le plugin
   d’analyse géospatiale actif. Vérifier que le compte a les autorisations de cet
   espace. Utiliser des conversations dédiées à la météo, à N1 et à B6.
2. Faire vérifier par l’opérateur l’installation et l’autorisation du paquet
   `fierce-pilots-2021-2024 / n1-local-20261014` et du cas
   `q3-2023-fos-native-recurrence / 2026.10.27.1` dans **cet** environnement.
   Un déploiement de code ne copie ni ces données ni leurs autorisations.
3. Exécuter une fois chaque scénario retenu. Noter les versions réellement
   servies, la date, l’environnement et les éventuelles limites. Les contrôles
   historiques de staging ne dispensent pas de cette répétition sur `.ai`.
4. Attendre la réponse finale et son enregistrement avant une relance ou un
   rechargement. Une vérification d’enregistrement en attente se réessaie avec
   le bouton prévu ; elle ne nécessite pas une nouvelle acquisition.
5. Préparer les onglets des preuves publiques cités dans chaque séquence.
   Masquer les adresses mail et données de compte à l’image.

Les commandes des scénarios A à E sont copiables telles quelles lorsque les
paquets indiqués sont installés. Les exemples comportant `<...>` sont des
gabarits : l’opérateur doit les compléter avec les identités réellement
autorisées avant le tournage.

Si un résultat est indisponible, conserver son statut réel. Corriger le
prérequis avant la captation ou montrer la preuve datée en l’annonçant comme
telle. Une capture de preuve historique ne devient pas une exécution du jour.

## 3. Montage principal proposé — environ 15 minutes

| Temps indicatif | Séquence | Message principal |
| --- | --- | --- |
| 0:00–0:45 | Contexte, environnement et outils | Un serveur MCP spécialisé, utilisé dans DEPOSIUM ; trois familles de données. |
| 0:45–4:15 | A — Trois pilotes, trois indicateurs, lecture et analyse descriptive | Les données sont accessibles ensemble, paginées et relisibles. |
| 4:15–6:15 | B — Météo, vues et seuils | Des valeurs fournisseur, des calculs explicites et le même résultat conservé. |
| 6:15–7:45 | C — NDVI satellite | Des acquisitions réelles, leurs dates et leurs limites. |
| 7:45–10:00 | D — Exemple B6 | Deux indices expliqués, avec les exclusions et les sources. |
| 10:00–12:15 | F — Preuves scientifiques | Ce qui a été comparé, reproduit et effectivement conclu. |
| 12:15–14:00 | G ou H — Classification ou surveillance | Référence climatologique, ou bulletin et seuils, selon le parcours préparé. |
| 14:00–15:00 | Synthèse et preuve complémentaire E | Accès contrôlé, résultat conservé, périmètre livré et maturité. |

G et H peuvent être deux clips annexes si leur préparation allonge le film.
Le minimum à conserver est : **`.ai` visible, trois pilotes, trois indicateurs
ensemble, analyse combinée documentée, relecture et distinction entre résultats
scientifiques et fonctionnement logiciel**.

## A. Trois pilotes et trois indicateurs historiques ensemble

**Cible : KPI 3 / B4, KPI 4 et KPI 7 ; conservation et accès autorisé.**
Preuves : [jointure des archives](proofs/proof-04.md),
[qualification N1](proofs/proof-11.md),
[analyse conjointe des pilotes](proofs/proof-23.md).

### A1. Ouvrir la première page

Dans une conversation dédiée :

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"n1-local-20261014","filters":{},"limit":100}
```

**Attendu pour ce paquet précis :** 130 correspondances, 100 lignes sur cette
page, continuation disponible. Montrer le tableau et les colonnes : zone,
jour ERA5 UTC, acquisition NDVI, écart de dates, NDVI, température et pluie.
Ouvrir les sources et la méthode : version, empreintes et limites font partie
du résultat, pas seulement le texte rédigé dans la réponse.

À dire : « Chaque ligne rapproche une observation de végétation et des données
météo historiques. Les trois indicateurs sont consultables ensemble. »

### A2. Lire la suite puis recharger

Cliquer **Page suivante sur le panneau scientifique de la réponse**. La
pagination interne d’un tableau peut seulement parcourir les lignes déjà
chargées : elle n’est pas la continuation du paquet.

Si ce bouton n’est pas disponible, copier **entièrement** la commande de
continuation présente dans la réponse, avec son curseur réel. Garder les mêmes
paquet, version, filtres et limite. Ne pas recopier un ancien curseur d’une
autre conversation.

**Attendu :** 30 lignes supplémentaires, toutes de Montpellier Métropole,
« Fin de la sélection », aucune page suivante. Les deux pages donnent
**100 + 30 = 130 lignes**, pas 130 jours consécutifs.

Recharger la page du navigateur, puis envoyer :

```text
scientific_readback
```

**Attendu :** relecture de la dernière page enregistrée, donc **30 lignes**
à ce stade. Mêmes valeurs, version et sources ; aucune nouvelle collecte.
Ce message ne demande pas une nouvelle agrégation des 130 lignes.

### A3. Présenter chaque pilote sans tronquer ses observations

Envoyer successivement :

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"n1-local-20261014","filters":{"zone_id":"bassin_du_lez"},"limit":100}
```

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"n1-local-20261014","filters":{"zone_id":"fos_sur_mer"},"limit":100}
```

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"n1-local-20261014","filters":{"zone_id":"montpellier_metropole"},"limit":100}
```

| Pilote | Lignes attendues | Période effectivement représentée |
| --- | ---: | --- |
| Bassin du Lez | 41 | 2021-02-11 → 2024-12-27 |
| Fos-sur-Mer | 48 | 2021-01-24 → 2024-12-29 |
| Montpellier Métropole | 41 | 2021-02-11 → 2024-12-27 |

Chaque réponse doit être terminale, sans troncature. Dans **Sources / méthode /
texte intégral**, montrer les moyennes calculées pour les trois variables et
leurs effectifs connus `n`. Les cellules absentes sont exclues ; les zéros sont
conservés. Avec ces filtres, une page contient toutes les observations retenues
du pilote. Sur une page globale tronquée, ces moyennes ne concerneraient que
les lignes de cette page.

### A4. Montrer une analyse combinée sans inventer une fonction du chat

Comparer les trois résumés précédents : NDVI, température et pluie sont
présents dans la même sélection, avec leurs effectifs. Reprendre les valeurs
**réellement retournées** dans un carton comparatif si nécessaire.

À dire : « Le service calcule des résumés descriptifs des observations retenues.
Ces moyennes décrivent l’échantillon aux dates d’acquisition. Elles ne sont pas
des moyennes climatiques de tous les jours de 2021 à 2024. »

Pour la partie **analyse conjointe au-delà de l’affichage**, ouvrir ensuite le
[paquet natif](../../src/data/pattern-library-2026-10-15/package/manifest.json)
et son [compte rendu](proofs/proof-23.md) : NDVI et six variables ERA5 ont été
analysés ensemble sur les trois mêmes pilotes. On obtient six bibliothèques,
**23 motifs GRAANK + 395 TGrad = 418 enregistrements**, tous non qualifiés.
La [comparaison WP0 reproduite](proofs/proof-26.md) constitue une seconde preuve
d’analyse conjointe sur ces données.

Présenter explicitement cette transition : « Voici l’analyse reproductible
réalisée sur les données ; sa preuve est distincte de leur lecture dans le
chat. » Ne pas faire passer un commentaire du LLM pour une corrélation
calculée, un nouveau minage ou une validation indépendante.

**Limites à laisser visibles :** NDVI sans unité ; température ERA5 moyenne
journalière en °C ; pluie ERA5 quotidienne en mm ; date d’acquisition distincte
du jour ERA5 ; appariement conservé au plus proche à ±3 jours. Pas
d’interpolation quotidienne. Lez et Métropole se chevauchent. Les géométries
historiques et points ERA5 ont leurs limites documentées.

## B. Prévision actuelle, visualisations et calcul sur les mêmes valeurs

**Cible : KPI 3 / B5 et intégration dans l’application.**
Références : [outils météo](../../reference/mcp-tools.md#weather-and-climate),
[périmètre d’interface](proofs/proof-18.md).

### B1. Acquérir une prévision explicite

Dans une nouvelle conversation, utiliser le formulaire **Prévisions météo**,
ou envoyer :

```text
geoai_weather {"version":1,"latitude":43.45455,"longitude":4.89795,"forecast_days":3,"variables":["precipitation","wind_speed","temperature_2m"]}
```

Il s’agit d’un **point de démonstration à Fos-sur-Mer**, pas d’un polygone de
commune. Le nom d’une ville ne prouve pas que le fournisseur emploie son
centre-ville. L’ancien point enregistré `montpellier` et le pilote
`montpellier_metropole` sont aussi des entrées différentes.

**Attendu :** dates UTC, coordonnées interrogées, valeurs, unités, horizon servi
et demandé, fournisseur/modèle effectivement utilisé, éventuel repli, licence
et collecte GeoAI. Les chiffres sont ceux du jour : aucune température ni
aucun dépassement n’est prédéterminé dans ce scénario.

Montrer la distinction : `issued_at` décrit ici la collecte GeoAI ; l’émission
du modèle reste inconnue si le fournisseur ne la fournit pas séparément.

### B2. Réutiliser la réponse enregistrée

Après enregistrement, envoyer ces messages l’un après l’autre :

```text
Génère un tableau à partir de ces mêmes données.
```

```text
Affiche un graphique à barres de ces prévisions.
```

```text
Montre ces zones sur une carte.
```

```text
Avec ces mêmes données, quels jours dépassent 20 mm de pluie ou 10 m/s de vent ? Donne les dates et les valeurs.
```

```text
D’où viennent ces données ?
```

**Attendu :** tableaux et graphiques issus de la prévision conservée, avec
variables et unités distinctes ; carte du point interrogé ; dates et valeurs
des dépassements stricts `>` ou absence de dépassement **sur les valeurs
évaluables**. Aucun nouvel appel météo pour ces relectures.

Les précipitations sont des cumuls par jour UTC ; le vent est le maximum
journalier à 10 m en m/s, **pas une rafale**. Une valeur manquante reste inconnue.
La carte ne représente pas une couche de pluie ni une emprise de pilote.

Ce lecteur de seuils conversationnel traite un ou deux seuils distincts de
pluie/vent, stricts ou inclusifs, avec **OU**. Ne pas lui attribuer la logique
plus étendue de l’agent de surveillance du scénario H.

### B3. Vérifier la continuité

Recharger, puis :

```text
Résume ces prévisions.
```

**Attendu :** mêmes valeurs et heure de collecte que B1. Une relance doit rester
dans la conversation où cette prévision est conservée. Les demandes ambiguës,
un changement de sujet ou une acquisition échouée peuvent exiger une nouvelle
requête explicite ; le scénario ne qualifie pas toutes les formulations
possibles d’une conversation libre.

Ce parcours prouve l’acquisition, la présentation et la réutilisation des
résultats. La **précision prévisionnelle** est évaluée séparément, en F.

## C. Satellite : profil NDVI et acquisitions réellement retenues

**Cible : KPI 3 / B3 et intégration satellite.**
Références : [NDVI livré](proofs/proof-03.md),
[interface et limites de qualification](proofs/proof-18.md).

Dans une nouvelle conversation, ouvrir **Profil NDVI satellite** :

| Champ | Valeur de démonstration |
| --- | --- |
| Ouest, sud, est, nord | `4.88`, `43.44`, `4.91`, `43.47` |
| Début / fin | `2026-09-01` / `2026-09-30` |
| Espacement minimum | `16` jours |
| Couverture nuageuse maximum | `20` % |

Commande équivalente :

```text
geoai_ndvi {"version":1,"bbox":[4.88,43.44,4.91,43.47],"start_date":"2026-09-01","end_date":"2026-09-30","interval_days":16,"cloud_cover_max":20}
```

**Attendu :** acquisitions effectivement trouvées et retenues, dates, identités
STAC, statistiques NDVI et nuages ; distinction entre scènes recherchées,
traitées avant espacement et observations retenues. Une consultation externe
du même mois ne garantit pas exactement les chiffres d’une ancienne réponse.

Sur ce parcours explicite, montrer le panneau du résultat conservé : tableau,
graphique de points et **Sources et méthode**. Le graphique conserve les
écarts temporels et ne relie pas les acquisitions pour suggérer une série
quotidienne. Sans acquisition retenue, montrer le statut. Sans moyenne NDVI
datée exploitable, aucun graphique n’est attendu.
Recharger pour vérifier le même résultat enregistré. En cas de
persistance en attente, utiliser **Réessayer la vérification enregistrée**.

À dire : « Le NDVI décrit la verdure de cette emprise aux dates observées. Nous
conservons les trous d’acquisition. Un signal de stress ne suffit pas à établir
sa cause, une perte de rendement ou une quantité de carbone. »

Le filtre est **nuages de la scène < 20 %**, pas un contrôle de qualité de chaque
pixel. L’intervalle est un minimum, pas une garantie de périodicité. Un minimum
NDVI négatif ne permet pas d’identifier eau, ombre ou artefact sans autre
contrôle. Le NDVI n’a pas d’unité.

La preuve publiée distingue les tests d’interface et le parcours connecté.
Cette répétition doit donc noter si la **nouvelle acquisition avec panneau
riche puis rechargement** a effectivement été observée dans l’environnement
filmé. Une requête en langage naturel peut fournir le texte sans emprunter ce
parcours de panneau conservé ; utiliser ici le formulaire ou la commande.

## D. B6 : comprendre un résultat explicatif historique

**Cible : KPI 3 / B6, calcul traçable et réception du périmètre adopté.**
Preuves : [cas historique reçu](proofs/proof-06.md),
[réception bornée](proofs/record-07.md).

Ouvrir **Exemple historique B6 → Consulter l’exemple**, ou envoyer :

```text
level2_historical_case {"case_id":"q3-2023-fos-native-recurrence","case_version":"2026.10.27.1"}
```

**Attendu pour ces octets et cette version :** 149 motifs lus, 147 exclus,
2 admissibles, dans cet ordre :

| Motif | Support | Récurrence | Accord | Décalage | Indice explicatif |
| --- | ---: | ---: | ---: | ---: | ---: |
| 39 | 0,6105 | 4/5, soit 0,8 | 1 | +2 jours | ≈ 0,42518 |
| 15 | 0,592 | 5/5, soit 1 | 1 | +5 jours | ≈ 0,41861 |

Déplier les facteurs :

```text
indice = support × stabilité de récurrence × accord directionnel × 0,5^(|décalage|/10)
```

Déplier les exclusions : **85 zone cible, 12 stabilité, 40 horizon,
10 valeurs prévues insuffisantes = 147**. Ces comptes utilisent le premier
filtre échoué, pas des catégories cumulatives.

Montrer ensuite les dates : émission GFS **2023-06-30 à 00:00 UTC**, températures
archivées du **1er au 5 juillet 2023**, collecte de l’archive bien plus tardive.
Le point historique Fos est **43.445, 4.92**, distinct du point de B1. Les valeurs
sont les moyennes de huit échantillons instantanés à pas de trois heures par
jour UTC ; humidité et ET0 restent absentes.

À dire : « Nous expliquons pourquoi ces deux motifs franchissent les filtres
d’ingénierie et comment leurs indices sont calculés. **0,425 n’est pas 42,5 % de
probabilité.** La récurrence entre zones n’est pas une précision prédictive.
Ce cas historique n’active aucune alerte opérationnelle. »

Recharger la conversation et rouvrir les sources/facteurs du panneau conservé.
Un échec de vérification doit rester visible ; ne pas certifier un score avant
la vérification de son enregistrement. La relecture ne récupère pas de météo
actuelle et ne relance pas l’extraction des motifs.

## E. Refus contrôlé, sans modifier les permissions réelles

**Cible : absence de données inventées et respect d’une version autorisée.**
Preuve historique : [refus et relectures N1](proofs/proof-11.md).

Dans une conversation de test séparée :

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"video-version-non-enregistree-20261031","filters":{},"limit":100}
```

**Attendu :** indisponibilité/refus ; aucune valeur du paquet demandé et aucune
collecte externe de remplacement. Le libellé affiché peut différer du code
interne. Vérifier au préalable que cette version de test n’est pas enregistrée.

Ce test démontre un **refus de version non enregistrée**. Il ne démontre pas à
lui seul toutes les séparations entre utilisateurs et espaces : les essais
contrôlés d’autorisation, de curseur altéré et d’empreinte modifiée ont leur
preuve distincte. Ne pas falsifier un identifiant d’utilisateur dans le chat.

## F. Montrer les résultats scientifiques avec leur vrai niveau de preuve

**Cible : KPI 2 / WP0–WP4 et évaluation de B5.** Ces séquences consistent à
ouvrir des preuves et artefacts existants. Elles ne relancent pas les études
dans le chat.

| Carton à montrer | Résultat à expliquer | Lien public |
| --- | --- | --- |
| Comparaison WP0 sur les données des trois pilotes | so4gp 1.1.2 / GRAANK contre mlxtend 0.25.0 / Apriori ; 20 identités partagées parmi 91 références, 71 différences expliquées, supports identiques pour les identités partagées ; reproduction Linux. Pas un classement universel des algorithmes. | [Preuve WP0](proofs/proof-26.md) |
| Bibliothèques natives | Six bibliothèques, 418 enregistrements ; tous non qualifiés. L’intégration les consomme et refuse une conclusion prédictive non justifiée. | [Capture native](proofs/proof-23.md), [qualification](proofs/proof-24.md) |
| Robustesse et validation | Seuils initiaux support ≥ 0,50 et confiance conditionnelle ≥ 0,70 : 0/10 candidats spatiaux passent au moins 4/5 plis ; 6/52 règles temporelles passent, soit trois paires inverses, qui échouent à Fos. | [Évaluation](proofs/proof-22.md) |
| Prévisions contre référence P50 | 86 040 paires par identité de modèle ; GFS positif dans 11/24 cellules agrégées de skill MAE, AIFS dans 22/24. Ce sont des cellules de comparaison, pas des taux de bonnes prévisions. | [Matrices B5](proofs/proof-05.md) |
| Test prédictif NDVI | Sur le test principal à Fos, ajouter les motifs ne réduit pas la MAE : ≈ 0,02263 → 0,02354 ; persistance ≈ 0,02237. Le résultat est négatif pour ce protocole. | [Ablation](proofs/proof-21.md) |
| Outil livré et appelé | Installation, contrats et appels du serveur ; interface applicative pour les restitutions, stockage et notifications. | [Livraison](proofs/proof-02.md), [guide des outils](../../reference/mcp-tools.md) |

Attention aux dénominateurs : **130 observations N1**, **418 motifs natifs**,
**179 motifs du catalogue Q3** et **149 TGrad dans B6** sont des objets
différents. Les deux motifs admissibles en D ne qualifient pas scientifiquement
les 418 motifs des trois pilotes.

Phrase de conclusion : « Nous avons livré l’accès aux données, l’exécution et
la traçabilité des calculs. Les essais de motifs sont reproductibles, mais
n’établissent pas de gain prédictif utilisable dans ce périmètre. Les alertes
sur des seuils météo restent un service distinct et fonctionnel. »

Si une restitution RAG est souhaitée, on peut placer une **copie autorisée de
l’annexe publique** dans l’espace et demander :

```text
Dans l’annexe technique FIERCE fournie, résume les résultats WP0, WP2 et du test prédictif NDVI. Cite les passages utilisés et distingue les résultats positifs, négatifs et non démontrés.
```

L’opérateur doit vérifier la réponse contre les passages sources avant le film.
Ce plan montre de la recherche documentaire ; il ne transforme pas le LLM en
évaluateur scientifique ni en exécuteur du benchmark.

## G. Option : comparer une prévision à des normales climatiques

**Cible : B6 niveau 1.** Preuve : [passage manuel réel](proofs/proof-13.md).

Ce parcours nécessite un **agent veilleur déjà configuré pour la veille
manuelle Site-Watch** par l’opérateur, dans le même espace et la conversation
ouverte. Un agent veilleur générique ne suffit pas.

Le profil historique démontré utilise `fos_sur_mer`, pluie et vent,
avec un horizon demandé de sept jours. Dans le détail de cet agent, cliquer
**Lancer**. Ce n’est ni une commande à coller dans le chat, ni le lancement
immédiat du cron du bulletin.

**Attendu si la configuration et les normales sont autorisées :** une réponse
« Veille manuelle », source météo, horizon réellement servi, référence
**1991–2020**, observations disponibles/manquantes/sèches, bandes et
classifications `notify`, `watch`, `none`. La pluie utilise le P90 des
**jours humides**, le vent sa distribution de référence ; afficher l’effectif.

Les valeurs et nombres de signaux dépendent de la prévision récupérée. Les
trois signaux du passage documenté du 4 octobre ne sont pas promis pour une
nouvelle exécution. Une classification `notify` ne prouve pas un mail envoyé ;
ce parcours manuel indique qu’aucun envoi n’a été demandé.

À dire : « Nous situons une prévision par rapport à une distribution historique
documentée. Il ne s’agit pas d’une vigilance officielle ni d’une compétence
prédictive calibrée. »

Sans agent manuel et normales configurés, montrer la preuve datée et le dire.
Une question du type « ces pluies sont-elles inhabituelles ? » posée au seul
lecteur de prévision conservée ne fournit pas les normales manquantes.

## H. Option : bulletin quotidien, surveillance et résultats transmis

**Cible : exploitation applicative et suivi des règles.** Preuves :
[bulletin et mail observé](proofs/proof-10.md),
[publication dans le chat](proofs/proof-16.md),
[bulletins et surveillance](proofs/proof-17.md).

Dans la liste des agents de l’espace, ouvrir la configuration de
**GeoAI weather bulletin / Bulletin GeoAI**. Montrer :

1. **Lieu du bulletin et des alertes** : point GPS ou lieu de référence.
2. **Bulletin quotidien** et son **Heure quotidienne (Europe/Paris)**.
3. **Surveillance des seuils**, son activation et sa **Fréquence des contrôles**.
4. **Ajouter une alerte** : par exemple pluie **au moins 20 mm**, puis température
   minimale **au plus 0 °C** pour une surveillance indicative du gel.
5. L’envoi à **l’adresse de mon compte**, et la destination de conversation si
   cette publication est configurée.

Les réglages et leur sauvegarde se préparent avec le propriétaire du compte
avant le tournage. Pour une preuve d’envoi, prévoir une exécution planifiée
réelle ou utiliser une trace historique datée. La préparation de ce conducteur
n’active aucune règle et n’envoie aucun message.

**Attendu :** heure du bulletin indépendante de la cadence des contrôles,
paramètres conservés après réouverture, puis trace de l’exécution réelle et
résultat du canal choisi. Usage quotidien activé, sans alerte configurée : le
bulletin quotidien reste possible. Un mail accepté par SMTP n’est pas à lui seul une
preuve de réception ; montrer la boîte destinataire si cette dernière est
revendiquée, en masquant les données personnelles.

Règles à expliquer :

- Les règles indépendantes se combinent par **OU**. Les groupes **ET**, lorsque
  proposés par la configuration active, exigent leurs conditions sur le
  **même jour UTC** ; les groupes se combinent par OU.
- Sept variables du pilote : température moyenne, température minimale,
  précipitations, humidité, évapotranspiration, vent et rayonnement solaire,
  avec leurs unités. Les comparateurs de ce panneau sont inclusifs ≥ et ≤.
- Une température minimale prévue ≤ 0 °C est un seuil choisi par l’utilisateur,
  pas une mesure locale ni une garantie de dommage aux cultures.
- La surveillance évite les renvois répétés d’un même dépassement ; une valeur
  manquante ne devient pas un retour certain sous le seuil.
- Un contrôle chaque heure ne garantit pas un nouveau calcul fournisseur
  chaque heure. Le pilote évalue des agrégats quotidiens, pas des rafales
  instantanées, une évolution minute par minute ou un cumul multi-jours.
- La publication en conversation dépend de sa configuration ; elle n’est pas
  automatiquement produite par l’activation du mail. Le NDVI n’est pas une
  variable de ce bulletin météo.

À dire : « L’application planifie l’acquisition, applique des règles explicites
et garde une trace du résultat. Ce service ne dépend pas d’une bibliothèque
de motifs scientifiquement qualifiée. »

## I. Complément éventuel : sites d’un classeur et données solaires

Cette capacité élargit les usages, mais n’est pas nécessaire à la preuve des
trois indicateurs FIERCE. Préparer un lot terminé et autorisé avant le film.
Les identités suivantes doivent être remplacées par celles du lot :

```text
solar_workbook {"action":"read","space_id":"<ESPACE_UUID>","lot_id":"<LOT_UUID>"}
```

**Attendu :** sites, identifiants conservés comme texte, coordonnées issues du
classeur, états par site, cellules d’origine, source des valeurs solaires,
dates UTC, cadence, unités et attribution. La lecture n’est pas une nouvelle
collecte. Une nouvelle collecte demande des paramètres, droits fournisseur et
contrôles propres ; ne pas l’improviser pendant la captation.

Ce résultat ne calcule pas à lui seul le rendement photovoltaïque, la
séquestration carbone ou la biodiversité.

## 4. Ce que la vidéo couvre, et ce qui reste dans les pièces justificatives

| Objectif technique | Séquences à filmer | Pièces complémentaires à conserver |
| --- | --- | --- |
| B1 : zones | A, carte du point en B avec sa portée explicite | [Périmètre des zones](proofs/proof-02.md) : 12 zones de panel + 3 pilotes ; la carte météo ne prouve pas leurs frontières. |
| B2 : données et normales ERA5 | A et G si configuré | [Couverture](proofs/proof-09.md), [matérialisation](proofs/proof-07.md) : 15 zones, six variables, 1991–2020 ; la vidéo de quelques lignes ne vérifie pas cette totalité. |
| B3 : Sentinel-2 / NDVI | A et C | [Archive et code](proofs/proof-03.md) ; les acquisitions de C ne prouvent pas à elles seules l’archive complète. |
| B4 : données unifiées | A | [Jointure et provenance](proofs/proof-04.md), [relecture](proofs/proof-11.md). |
| B5 : prévisions et comparaison | B et F | [Matrices, couverture et modèles](proofs/proof-05.md). Aucun gain n’est déduit d’une capture de prévision. |
| B6 : deux niveaux d’analyse | D et G | [Réception du périmètre explicatif](proofs/proof-06.md) ; aucune confiance calibrée ajoutée. |
| WP0 à WP3 : comparaison, motifs, évaluation | A4 et F | [WP0](proofs/proof-26.md), [capture](proofs/proof-23.md), [validation](proofs/proof-22.md), [récurrence Q3](proofs/proof-12.md), [qualification](proofs/proof-24.md). |
| WP4 : livraison logicielle | Intro et parcours fonctionnels | [Packaging et périmètre adopté](proofs/proof-02.md), [traçabilité so4gp](proofs/proof-25.md), contrats et guide MCP. La vidéo ne mesure pas une taille d’image ARM64. |
| KPI 4 / KPI 7 : trois indicateurs, trois pilotes, analyse combinée | **A1–A4 indispensables** | [Correspondance des indicateurs](fierce-kpi-evidence.en.md#6-kpi-4--three-indicators-across-the-three-pilots). Sentinel-2 NDVI + deux grandeurs ERA5 : pas trois mesures satellitaires indépendantes. |
| Intégration et maturité | A à H selon configuration | [Limite applicative](proofs/proof-14.md), [évaluation TRL](fierce-kpi-evidence.en.md#9-objective-maturity-assessment-and-trl-evidence). TRL 6 autoévalué du démonstrateur intégré ; TRL 7 non établi ; prédiction nouvelle par motifs non validée. |

Les adaptations de périmètre doivent rester explicites : AIFS et repli physique
à la place du comparatif FourCastNet initial, 30 ans d’ERA5 livrés, skill mesuré
jusqu’à J7 et indice explicatif au lieu d’une confiance probabiliste calibrée.
Voir les [correspondances avec le plan](fierce-kpi-evidence.en.md#5-kpi-3--d3--the-six-geospatial-pipeline-modules).
La vidéo ne doit pas promettre une qualité validée jusqu’à J16.

Les autres KPI d’impact n’appellent pas un chiffre déduit de cette vidéo :
aucune économie mesurée de ressources, de pollution ou de carbone n’est établie
par l’affichage d’une alerte. L’étude de marché est hors de ce conducteur.

## 5. Trace à remplir après la répétition et après le tournage

Conserver cette fiche avec la vidéo. La version publique ne doit contenir
ni secrets, ni lien de conversation privée, ni adresse personnelle.

| Élément | À renseigner avec l’observation réelle |
| --- | --- |
| Date et environnement | Date/heure avec fuseau ; `.ai` ou nom réel de la répétition. |
| Versions | Révisions du serveur GeoAI et du client applicatif utilisées ; versions des paquets/cas. |
| A — Données | 100 + 30 ; filtres 41 / 48 / 41 ; trois indicateurs ; relecture après rechargement. |
| A4 / F — Analyse | Liens des artefacts effectivement montrés ; calculs descriptifs et analyse de motifs distingués. |
| B — Prévision | Coordonnées, dates servies, source, heure de collecte et résultat des seuils ; valeurs conservées après rechargement. |
| C — NDVI | Emprise, dates, nombre réel d’acquisitions ; panneau riche et rechargement effectivement observés ou non. |
| D — B6 | Version ; 149 / 147 / 2 ; indices ; sources et vérification du résultat conservé. |
| E — Refus | Version de test non enregistrée ; aucun remplacement ni valeur inventée. |
| G / H — Surveillance | Parcours réellement exécuté, ou preuve historique datée ; trace et réception séparées. |
| Vidéo | URL finale et minutages des séquences ; éventuelles coupes annoncées. |
| Écarts | Incident, absence d’autorisation, donnée manquante, fonction non montrée et pièce de remplacement identifiée. |

La captation prouve les gestes et résultats observés dans son environnement.
Les preuves scientifiques et les réceptions internes restent les documents
liés ; elles ne deviennent pas une approbation externe de FIERCE.
