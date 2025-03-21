pgn_file=$1

echo "readpgn ${pgn_file}
elo
    mm
    offset 2750 StockfishPlayer-2750
    ratings >ratings
    x
x

" > bayeselo_ratings_script