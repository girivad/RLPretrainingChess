pgn_file=$1

echo "readpgn ${pgn_file}
elo
    mm
    offset 1350 Stockfish-1350
    ratings >ratings
    x
x

" > bayeselo_ratings_script