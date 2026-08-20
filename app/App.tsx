import React from 'react'
import { StyleSheet, Text, View } from 'react-native'

export default function App(): React.JSX.Element {
  return (
    <View style={styles.экран}>
      <Text style={styles.текст}>CarLens M-1.6</Text>
    </View>
  )
}

const styles = StyleSheet.create({
  экран: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  текст: { color: '#fff', fontSize: 18 },
})
