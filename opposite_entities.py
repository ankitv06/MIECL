# opposite_entities.py
# Curated WikidataId -> opposite WikidataId mapping
# Only covers entities from 'news' and 'sports' categories with meaningful polar opposites
# Entity types covered: P (Person), O (Organization), G (Geography)

ELIGIBLE_CATEGORIES = {'news', 'sports'}
ELIGIBLE_TYPES = {'P', 'O', 'G'}

OPPOSITE_ENTITY_MAP = {

    # ── Politics: People ──────────────────────────────────────────────────────
    'Q22686':   'Q6279',      # Donald Trump      ↔ Joe Biden
    'Q6279':    'Q22686',     # Joe Biden          ↔ Donald Trump
    'Q212648':  'Q934898',    # Rudy Giuliani      ↔ Elijah Cummings
    'Q934898':  'Q212648',    # Elijah Cummings    ↔ Rudy Giuliani

    # ── Politics: Parties ────────────────────────────────────────────────────
    'Q29552':   'Q29468',     # Democratic Party   ↔ Republican Party
    'Q29468':   'Q29552',     # Republican Party   ↔ Democratic Party

    # ── Geography: Countries ─────────────────────────────────────────────────
    'Q212':     'Q159',       # Ukraine            ↔ Russia
    'Q159':     'Q212',       # Russia             ↔ Ukraine
    'Q30':      'Q148',       # United States      ↔ China
    'Q148':     'Q30',        # China              ↔ United States

    # ── NFL Teams (rival pairs) ───────────────────────────────────────────────
    'Q193390':  'Q223522',    # New England Patriots   ↔ Kansas City Chiefs
    'Q223522':  'Q193390',    # Kansas City Chiefs     ↔ New England Patriots

    'Q223527':  'Q191477',    # Cleveland Browns       ↔ Pittsburgh Steelers
    'Q191477':  'Q223527',    # Pittsburgh Steelers    ↔ Cleveland Browns

    'Q204862':  'Q219714',    # Dallas Cowboys         ↔ Philadelphia Eagles
    'Q219714':  'Q204862',    # Philadelphia Eagles    ↔ Dallas Cowboys

    'Q213837':  'Q205033',    # Green Bay Packers      ↔ Chicago Bears
    'Q205033':  'Q213837',    # Chicago Bears          ↔ Green Bay Packers

    'Q276539':  'Q223511',    # Baltimore Ravens       ↔ Cincinnati Bengals
    'Q223511':  'Q276539',    # Cincinnati Bengals     ↔ Baltimore Ravens

    'Q223507':  'Q223522',    # Denver Broncos         ↔ Kansas City Chiefs
    # Note: Q223522 already maps back to Q193390 (Patriots), so Denver's reverse
    # is handled by the Chiefs entry above

    'Q212654':  'Q204862',    # Washington Redskins    ↔ Dallas Cowboys

    'Q221878':  'Q337758',    # Seattle Seahawks       ↔ San Francisco 49ers
    'Q337758':  'Q221878',    # San Francisco 49ers    ↔ Seattle Seahawks

    'Q223243':  'Q193390',    # Miami Dolphins         ↔ New England Patriots

    'Q320476':  'Q272059',    # Tampa Bay Buccaneers   ↔ Atlanta Falcons
    'Q272059':  'Q320476',    # Atlanta Falcons        ↔ Tampa Bay Buccaneers

    'Q330120':  'Q172435',    # Carolina Panthers      ↔ New Orleans Saints
    'Q172435':  'Q330120',    # New Orleans Saints     ↔ Carolina Panthers

    'Q272223':  'Q320484',    # Jacksonville Jaguars   ↔ Tennessee Titans
    'Q320484':  'Q272223',    # Tennessee Titans       ↔ Jacksonville Jaguars

    'Q193753':  'Q223514',    # Indianapolis Colts     ↔ Houston Texans
    'Q223514':  'Q193753',    # Houston Texans         ↔ Indianapolis Colts

    'Q221150':  'Q213837',    # Minnesota Vikings      ↔ Green Bay Packers

    'Q337377':  'Q337758',    # Los Angeles Rams       ↔ San Francisco 49ers

    # ── NBA Teams ─────────────────────────────────────────────────────────────
    'Q121783':  'Q131371',    # Los Angeles Lakers     ↔ Boston Celtics
    'Q131371':  'Q121783',    # Boston Celtics         ↔ Los Angeles Lakers

    'Q161345':  'Q157376',    # Houston Rockets        ↔ Golden State Warriors
    'Q157376':  'Q161345',    # Golden State Warriors  ↔ Houston Rockets

    'Q164177':  'Q121783',    # Phoenix Suns           ↔ Los Angeles Lakers

    # ── MLB Teams ─────────────────────────────────────────────────────────────
    'Q848117':  'Q825838',    # Houston Astros         ↔ Washington Nationals
    'Q825838':  'Q848117',    # Washington Nationals   ↔ Houston Astros
}
